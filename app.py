import streamlit as st
import validators
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# LangChain v1.0 moved EnsembleRetriever out of the core `langchain`
# package into `langchain-classic`. Add "langchain-classic" to
# requirements.txt if you don't already have it installed.
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import re
import os
import hashlib
import time
import threading
from collections import deque
from bs4 import BeautifulSoup
import requests


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="LangChain RAG App",
    page_icon="🦜"
)

st.title("🦜 LangChain: RAG Q&A From YT or Website")


# ================= API KEY HANDLING =================
# Preferred: set GROQ_API_KEY in Streamlit secrets (.streamlit/secrets.toml
# locally, or the "Secrets" panel on Streamlit Cloud/your host). Falls back
# to manual entry so the app still works for anyone without secrets set up.

def get_api_key_from_secrets():
    # Streamlit Community Cloud stores secrets in st.secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        # No secrets.toml present at all — fine, fall through
        pass

    # Render / Railway / plain Docker inject secrets as env vars instead
    return os.environ.get("GROQ_API_KEY")


with st.sidebar:

    secret_api_key = get_api_key_from_secrets()

    if secret_api_key:

        groq_api_key = secret_api_key

        st.success("Groq API Key loaded from secrets ✅")

    else:

        groq_api_key = st.text_input(
            "Enter Groq API Key",
            type="password",
            help="No key found in secrets — enter one manually for this session."
        )


# ================= RATE LIMITING =================
# Basic in-memory rate limiting to protect the shared Groq API budget
# from being drained by heavy or automated use.
#
# LIMITATION: this is per-process, in-memory state. It works correctly
# for a single running container. If you ever scale to multiple replicas
# behind a load balancer, each replica has its own counters — for a real
# distributed limit at that point, use a shared store (e.g. Redis) or
# rate limiting at the reverse-proxy/CDN layer instead.

RATE_LIMIT_LOCK = threading.Lock()
GLOBAL_REQUEST_LOG = deque()   # timestamps of every LLM call, all users combined

GLOBAL_MAX_REQUESTS = 30       # max LLM calls across ALL users per window
GLOBAL_WINDOW_SECONDS = 60

SESSION_MAX_REQUESTS = 5       # max LLM calls per individual browser session
SESSION_WINDOW_SECONDS = 60


def _prune_old(timestamps, window_seconds):
    cutoff = time.time() - window_seconds
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()


def check_rate_limit():
    """
    Call this right before any expensive/paid operation (embedding
    generation or an LLM call). Returns (allowed: bool, message: str).
    Enforces both a per-session limit (stops one user from over-using
    their own access) and a global limit (protects total API spend).
    """

    now = time.time()

    # ---------- per-session check ----------

    if "request_timestamps" not in st.session_state:
        st.session_state.request_timestamps = deque()

    session_ts = st.session_state.request_timestamps
    _prune_old(session_ts, SESSION_WINDOW_SECONDS)

    if len(session_ts) >= SESSION_MAX_REQUESTS:
        return False, (
            f"You've hit the limit of {SESSION_MAX_REQUESTS} requests per "
            f"{SESSION_WINDOW_SECONDS} seconds. Please wait a moment and try again."
        )

    # ---------- global check (shared across all users of this instance) ----------

    with RATE_LIMIT_LOCK:

        _prune_old(GLOBAL_REQUEST_LOG, GLOBAL_WINDOW_SECONDS)

        if len(GLOBAL_REQUEST_LOG) >= GLOBAL_MAX_REQUESTS:
            return False, (
                "This app is receiving high traffic right now. "
                "Please try again in a minute."
            )

        GLOBAL_REQUEST_LOG.append(now)

    session_ts.append(now)

    return True, ""


# ================= URL INPUT =================

generic_url = st.text_input(
    "Enter YouTube or Website URL"
)


# ================= EXTRACT YOUTUBE VIDEO ID =================

def extract_video_id(url):

    # Each pattern anchors on an actual YouTube URL shape (watch/shorts/
    # embed/youtu.be), not just "any 11 characters after a slash" — the
    # old version incorrectly matched non-YouTube URLs whose path
    # happened to be 11 characters long (e.g. "/not-a-video").
    patterns = [
        r"youtube\.com/watch\?v=([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"youtube\.com/embed/([0-9A-Za-z_-]{11})",
        r"youtube\.com/shorts/([0-9A-Za-z_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


# ================= LOAD CONTENT =================

# ================= WEBSITE CONTENT CLEANER =================
# Raw page scraping (e.g. plain WebBaseLoader) grabs EVERY visible string
# on the page — nav menus, sidebars, "Edit"/"View history" links, footers,
# etc. On sites like Wikipedia this boilerplate can be as long as the
# actual article, and it pollutes both the summary and the embeddings.
# This strips known-boilerplate tags and prefers semantic content
# containers (<article>, <main>) when the page provides them.

def clean_website_text(url, timeout=10):

    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAGSummarizerBot/1.0)"}
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove tags that are essentially never part of the actual content
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
    ):
        tag.decompose()

    # Remove common boilerplate containers by id/class, seen across many
    # sites (Wikipedia's sidebar/toolbox, cookie banners, etc.)
    boilerplate_selectors = [
        {"id": "mw-panel"},                # Wikipedia sidebar
        {"id": "footer"},
        {"class": "vector-page-toolbar"},  # Wikipedia page tools
        {"class": "navbox"},
        {"class": "cookie-banner"},
    ]
    for selector in boilerplate_selectors:
        for tag in soup.find_all(attrs=selector):
            tag.decompose()

    # Prefer a semantic main-content container if the page has one
    main_content = (
        soup.find(id="mw-content-text")   # Wikipedia article body
        or soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )

    text = main_content.get_text(separator=" ", strip=True)

    # Collapse repeated whitespace left behind after stripping tags
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError(
            "Could not extract readable content from this page."
        )

    return text


def load_content(url):

    # ---------- YOUTUBE ----------

    if "youtube.com" in url or "youtu.be" in url:

        video_id = extract_video_id(url)

        if not video_id:
            raise ValueError(
                "Could not extract a video ID from this YouTube URL. "
                "Please check the link and try again."
            )

        try:
            transcript = YouTubeTranscriptApi().fetch(video_id)
        except TranscriptsDisabled:
            raise ValueError("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            raise ValueError("No transcript is available for this video.")
        except VideoUnavailable:
            raise ValueError("This video is unavailable or private.")

        text = " ".join(
            [item.text for item in transcript]
        )

    # ---------- WEBSITE ----------

    else:

        try:
            text = clean_website_text(url)
        except requests.exceptions.Timeout:
            raise ValueError("The website took too long to respond.")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Could not fetch this page: {e}")

    return text


# ================= CACHED RESOURCES =================
# @st.cache_resource ensures these are created ONCE per session
# (or once per unique argument set) instead of on every button click.

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm(api_key):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key
    )


# ================= VECTOR STORE PERSISTENCE =================

VECTORSTORE_DIR = "vectorstore_cache"


def get_store_path(url):
    """Each URL gets its own folder, named by a hash of the URL."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    return os.path.join(VECTORSTORE_DIR, url_hash)


def create_vector_store(text, url):

    store_path = get_store_path(url)
    embeddings = get_embeddings()

    # ---------- REUSE EXISTING INDEX IF WE'VE SEEN THIS URL BEFORE ----------

    if os.path.exists(store_path):
        return FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # ---------- OTHERWISE BUILD AND SAVE A NEW ONE ----------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    vectorstore = FAISS.from_texts(
        chunks,
        embeddings
    )

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(store_path)

    return vectorstore


# ================= HYBRID RETRIEVER =================
# Pure vector similarity misses exact keyword matches (names, numbers,
# acronyms). BM25 is a classic keyword-ranking algorithm that's strong
# where embeddings are weak. Combining both ("hybrid search") covers
# more ground than either alone.

def get_hybrid_retriever(vectorstore, k=3):

    # All chunks are already stored inside the FAISS docstore —
    # reuse them to build the BM25 index instead of re-splitting text.
    all_docs = list(vectorstore.docstore._dict.values())

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    # weights: leans slightly toward semantic (vector) search while
    # still letting exact keyword hits from BM25 surface.
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )

    return hybrid_retriever


# ================= PROCESS URL =================

if st.button("Process URL"):

    # ---------- API KEY CHECK ----------

    if not groq_api_key:

        st.error("Please enter Groq API Key")

        st.stop()

    # ---------- URL VALIDATION ----------

    if not validators.url(generic_url):

        st.error("Please enter a valid URL")

        st.stop()

    # ---------- RATE LIMIT CHECK ----------

    allowed, rate_limit_message = check_rate_limit()

    if not allowed:

        st.error(rate_limit_message)

        st.stop()

    try:

        # ---------- LOAD CONTENT ----------

        with st.spinner("Loading content..."):

            text = load_content(generic_url)

            # Optional limit — warn instead of silently dropping content
            original_length = len(text)
            text = text[:15000]

            if original_length > 15000:
                st.warning(
                    f"Content was {original_length:,} characters long and has "
                    f"been truncated to the first 15,000 characters. Some "
                    f"information may be missing from summaries and answers."
                )

        # ---------- CREATE VECTOR STORE ----------

        with st.spinner("Creating Vector Store..."):

            vectorstore = create_vector_store(text, generic_url)

            st.session_state.vectorstore = vectorstore

        # ---------- INITIALIZE LLM ----------

        llm = get_llm(groq_api_key)

        # ---------- SUMMARY PROMPT ----------

        summary_prompt = f"""
        Summarize the following content clearly
        in around 150 words.

        Content:
        {text[:4000]}
        """

        # ---------- GENERATE SUMMARY ----------

        with st.spinner("Generating Summary..."):

            summary_response = llm.invoke(
                summary_prompt
            )

            st.session_state.summary = (
                summary_response.content
            )

        # ---------- SUCCESS ----------

        st.success("RAG Pipeline Ready!")

    except ValueError as e:

        st.error(str(e))

    except Exception as e:

        st.error("An error occurred while processing this URL.")

        st.exception(e)


# ================= SHOW SUMMARY =================

if "summary" in st.session_state:

    st.subheader("Summary")

    st.write(st.session_state.summary)


# ================= QUESTION INPUT =================

question = st.text_input(
    "Ask Questions From The Content"
)


# ================= QUESTION ANSWERING =================

if st.button("Ask Question"):

    # ---------- VECTOR STORE CHECK ----------

    if "vectorstore" not in st.session_state:

        st.error("Please process a URL first")

        st.stop()

    # ---------- EMPTY QUESTION CHECK ----------

    if not question:

        st.error("Please enter a question")

        st.stop()

    # ---------- RATE LIMIT CHECK ----------

    allowed, rate_limit_message = check_rate_limit()

    if not allowed:

        st.error(rate_limit_message)

        st.stop()

    try:

        vectorstore = st.session_state.vectorstore

        # ---------- RETRIEVE RELEVANT CHUNKS (HYBRID: BM25 + VECTOR) ----------

        retriever = get_hybrid_retriever(vectorstore, k=3)

        docs = retriever.invoke(question)

        # ---------- CREATE CONTEXT ----------

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # ---------- INITIALIZE LLM ----------

        llm = get_llm(groq_api_key)

        # ---------- QA PROMPT ----------

        prompt = f"""
        Answer the question using only the
        provided context.

        Context:
        {context}

        Question:
        {question}
        """

        # ---------- GENERATE ANSWER ----------

        with st.spinner("Generating Answer..."):

            response = llm.invoke(prompt)

        # ---------- STORE ANSWER + SOURCES ----------

        st.session_state.answer = response.content
        st.session_state.sources = docs

    except Exception as e:

        st.error("Error while generating answer")

        st.exception(e)


# ================= SHOW ANSWER =================

if "answer" in st.session_state:

    st.subheader("Answer")

    st.write(st.session_state.answer)

    # ---------- SHOW SOURCE CHUNKS ----------

    if "sources" in st.session_state and st.session_state.sources:

        with st.expander("View source chunks used for this answer"):

            for i, doc in enumerate(st.session_state.sources, start=1):

                st.markdown(f"**Chunk {i}**")

                st.write(doc.page_content)

                st.divider()