"""
Basic tests for the pure/testable functions in app.py.

Streamlit apps mix UI code (st.button, st.text_input) with logic, so we
only test the logic functions here — extract_video_id and get_store_path.
Testing st.* calls directly requires the streamlit AppTest framework and
is out of scope for a first CI pass.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import extract_video_id, get_store_path, _prune_old, clean_website_text
from collections import deque
import time
from unittest.mock import patch, Mock


class TestExtractVideoId:

    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_invalid_url_returns_none(self):
        url = "https://example.com/not-a-video"
        assert extract_video_id(url) is None

    def test_empty_string_returns_none(self):
        assert extract_video_id("") is None

    def test_shorts_url(self):
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_non_youtube_url_with_11_char_path_returns_none(self):
        # Regression test: an earlier regex incorrectly matched any
        # 11-character path segment on ANY domain, not just YouTube.
        url = "https://example.com/abcdefghijk"  # "abcdefghijk" is 11 chars
        assert extract_video_id(url) is None


class TestGetStorePath:

    def test_same_url_gives_same_path(self):
        url = "https://example.com/article"
        assert get_store_path(url) == get_store_path(url)

    def test_different_urls_give_different_paths(self):
        path_a = get_store_path("https://example.com/a")
        path_b = get_store_path("https://example.com/b")
        assert path_a != path_b

    def test_path_is_inside_cache_dir(self):
        path = get_store_path("https://example.com/article")
        assert "vectorstore_cache" in path


class TestPruneOld:

    def test_removes_timestamps_older_than_window(self):
        now = time.time()
        timestamps = deque([now - 100, now - 90, now - 5])
        _prune_old(timestamps, window_seconds=60)
        assert list(timestamps) == [now - 5]

    def test_keeps_all_timestamps_within_window(self):
        now = time.time()
        timestamps = deque([now - 10, now - 5, now - 1])
        _prune_old(timestamps, window_seconds=60)
        assert len(timestamps) == 3

    def test_empty_deque_stays_empty(self):
        timestamps = deque()
        _prune_old(timestamps, window_seconds=60)
        assert len(timestamps) == 0


class TestCleanWebsiteText:

    @patch("app.requests.get")
    def test_strips_navigation_boilerplate(self, mock_get):
        mock_response = Mock()
        mock_response.text = """
            <html><body>
            <nav id="mw-panel"><ul><li>Tools</li><li>Edit</li></ul></nav>
            <div id="mw-content-text"><p>Real article content here.</p></div>
            <div id="footer">Footer junk</div>
            </body></html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = clean_website_text("https://en.wikipedia.org/wiki/Test")

        assert "Real article content here." in result
        assert "Tools" not in result
        assert "Edit" not in result
        assert "Footer junk" not in result

    @patch("app.requests.get")
    def test_raises_value_error_on_empty_page(self, mock_get):
        mock_response = Mock()
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError):
            clean_website_text("https://example.com/empty")