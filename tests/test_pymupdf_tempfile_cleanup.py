"""Regression test: PyMuPDFScraper must not leak its downloaded temp file.

When scraping a remote PDF, the scraper downloads it to a
``NamedTemporaryFile(delete=False, suffix=".pdf")`` and then loads it with
``PyMuPDFLoader``. The old code called ``os.remove(temp_filename)`` only on the
success path, so a parse failure (malformed/partial PDF -> ``PyMuPDFLoader.load``
raises) left the temp file behind on disk every time. The exception is then
swallowed by the broad ``except``, so the leak was silent.
"""

import os
import glob
import tempfile
from unittest.mock import MagicMock, patch

import requests

from gpt_researcher.scraper.pymupdf.pymupdf import PyMuPDFScraper


class _FakeResponse:
    headers = {}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        yield b"%PDF-1.4 not-a-real-pdf"


def _temp_pdfs() -> set:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "*.pdf")))


def test_tempfile_removed_when_loader_raises():
    scraper = PyMuPDFScraper("https://example.com/broken.pdf")

    before = _temp_pdfs()

    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.requests.get",
        return_value=_FakeResponse(),
    ), patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.PyMuPDFLoader"
    ) as mock_loader:
        mock_loader.return_value.load.side_effect = RuntimeError("corrupt PDF")

        content, images, title = scraper.scrape()

    # Broad except still yields the empty-result contract...
    assert (content, images, title) == ("", [], "")
    # ...but no new *.pdf temp file is left behind.
    leaked = _temp_pdfs() - before
    assert not leaked, f"PyMuPDFScraper leaked temp file(s): {leaked}"


def test_pdf_timeout_returns_typed_failure():
    scraper = PyMuPDFScraper(
        "https://example.com/slow.pdf",
        connect_timeout_seconds=3,
        total_timeout_seconds=8,
    )
    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.requests.get",
        side_effect=requests.exceptions.Timeout("read timed out"),
    ):
        assert scraper.scrape() == ("", [], "")
    assert scraper.last_error_type == "download_timeout"
    assert "8s" in scraper.last_error_detail


def test_pdf_content_length_limit_returns_typed_failure():
    response = _FakeResponse()
    response.headers = {"Content-Length": str(1025)}
    scraper = PyMuPDFScraper(
        "https://example.com/large.pdf",
        max_download_bytes=1024,
    )
    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.requests.get",
        return_value=response,
    ):
        assert scraper.scrape() == ("", [], "")
    assert scraper.last_error_type == "size_limit"


def test_ssl_retry_uses_one_bounded_retry_path():
    response = _FakeResponse()
    request = MagicMock(
        side_effect=[requests.exceptions.SSLError("bad cert"), response]
    )
    scraper = PyMuPDFScraper(
        "https://example.com/retry.pdf",
        total_timeout_seconds=8,
    )
    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.requests.get",
        request,
    ), patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.PyMuPDFLoader"
    ) as loader:
        loader.return_value.load.return_value = []
        scraper.scrape()
    assert request.call_count == 2
    assert request.call_args_list[0].kwargs["verify"] is True
    assert request.call_args_list[1].kwargs["verify"] is False


def test_streaming_download_obeys_total_deadline_without_sleeping():
    response = _FakeResponse()
    scraper = PyMuPDFScraper(
        "https://example.com/stalled.pdf",
        total_timeout_seconds=8,
    )
    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.requests.get",
        return_value=response,
    ), patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.time.monotonic",
        side_effect=[0.0, 0.0, 8.1],
    ):
        assert scraper.scrape() == ("", [], "")
    assert scraper.last_error_type == "download_timeout"
