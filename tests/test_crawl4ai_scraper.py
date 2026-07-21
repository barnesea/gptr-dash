import unittest
from unittest.mock import patch

from gpt_researcher.scraper import Crawl4AIScraper
from gpt_researcher.scraper.arxiv.arxiv import ArxivScraper
from gpt_researcher.scraper.pymupdf.pymupdf import PyMuPDFScraper
from gpt_researcher.scraper.scraper import Scraper
from gpt_researcher.utils.workers import WorkerPool


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload=None, status_error=None):
        self.payload = payload or {}
        self.status_error = status_error
        self.calls = []

    def post(self, url, json, timeout, headers=None):
        self.calls.append(
            {"url": url, "json": json, "timeout": timeout, "headers": headers}
        )
        return FakeResponse(self.payload, self.status_error)


class Crawl4AITests(unittest.TestCase):
    def test_crawl4ai_scraper_returns_markdown_and_title(self):
        session = FakeSession(
            {
                "markdown": {"fit_markdown": "# Heading\n\n" + "Useful content " * 20},
                "metadata": {"title": "Crawl4AI Title"},
            }
        )

        scraper = Crawl4AIScraper("https://example.com", session)
        content, image_urls, title = scraper.scrape()

        self.assertIn("Useful content", content)
        self.assertEqual(title, "Crawl4AI Title")
        self.assertEqual(image_urls, [])
        self.assertEqual(session.calls[0]["url"], "http://crawl4ai:11235/md")
        self.assertEqual(
            session.calls[0]["json"],
            {"url": "https://example.com", "f": "fit"},
        )

    def test_crawl4ai_scraper_uses_configurable_endpoint_and_filter(self):
        session = FakeSession({"markdown": "Raw markdown " * 20})

        with patch.dict(
            "os.environ",
            {
                "CRAWL4AI_BASE_URL": "http://crawler:11235/",
                "CRAWL4AI_FILTER": "raw",
                "CRAWL4AI_TIMEOUT_S": "12",
            },
        ):
            scraper = Crawl4AIScraper("https://example.com", session)
            scraper.scrape()

        self.assertEqual(session.calls[0]["url"], "http://crawler:11235/md")
        self.assertEqual(session.calls[0]["json"]["f"], "raw")
        self.assertEqual(session.calls[0]["timeout"], 12.0)

    def test_crawl4ai_scraper_sends_bearer_token_when_configured(self):
        session = FakeSession({"markdown": "Raw markdown " * 20})

        with patch.dict("os.environ", {"CRAWL4AI_API_TOKEN": "token-123"}):
            scraper = Crawl4AIScraper("https://example.com", session)
            scraper.scrape()

        self.assertEqual(
            session.calls[0]["headers"],
            {"Authorization": "Bearer token-123"},
        )

    def test_crawl4ai_scraper_extracts_title_from_markdown(self):
        session = FakeSession({"markdown": "# Markdown Title\n\n" + "Useful content " * 20})

        scraper = Crawl4AIScraper("https://example.com", session)
        _, _, title = scraper.scrape()

        self.assertEqual(title, "Markdown Title")

    def test_failed_crawl4ai_request_returns_empty_content(self):
        session = FakeSession(
            {"markdown": "ignored"},
            status_error=RuntimeError("blocked"),
        )

        scraper = Crawl4AIScraper("https://example.com", session)
        self.assertEqual(scraper.scrape(), ("", [], ""))

    def test_short_crawl4ai_content_is_dropped_without_crashing_batch(self):
        worker_pool = WorkerPool(max_workers=1)

        try:
            scraper = Scraper(
                ["https://example.com"],
                "test-agent",
                "crawl4ai",
                worker_pool=worker_pool,
            )
            scraper_class = scraper.get_scraper("https://example.com")
            fake_scraper = scraper_class(
                "https://example.com",
                FakeSession({"markdown": "short", "metadata": {"title": "Too Short"}}),
            )

            self.assertEqual(fake_scraper.scrape(), ("short", [], "Too Short"))
        finally:
            worker_pool.executor.shutdown(wait=False)

    def test_scraper_router_selects_crawl4ai_but_preserves_special_routes(self):
        worker_pool = WorkerPool(max_workers=1)

        try:
            scraper = Scraper(
                ["https://example.com"],
                "test-agent",
                "crawl4ai",
                worker_pool=worker_pool,
            )

            self.assertIs(scraper.get_scraper("https://example.com"), Crawl4AIScraper)
            self.assertIs(
                scraper.get_scraper("https://example.com/file.pdf"),
                PyMuPDFScraper,
            )
            self.assertIs(
                scraper.get_scraper("https://arxiv.org/abs/1234.5678"),
                ArxivScraper,
            )
        finally:
            worker_pool.executor.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
