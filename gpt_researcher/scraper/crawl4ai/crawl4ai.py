from __future__ import annotations

import os
import re
from typing import Any

import requests


class Crawl4AIScraper:
    """Scrape a URL with an external Crawl4AI service."""

    def __init__(self, link: str, session: requests.Session | None = None):
        self.link = link
        self.session = session or requests.Session()
        self.base_url = os.getenv("CRAWL4AI_BASE_URL", "http://crawl4ai:11235").rstrip("/")
        self.content_filter = os.getenv("CRAWL4AI_FILTER", "fit")
        self.timeout = float(os.getenv("CRAWL4AI_TIMEOUT_S", "90"))
        self.max_content_chars = int(os.getenv("CRAWL4AI_MAX_CONTENT_CHARS", "0") or 0)

    def scrape(self) -> tuple[str, list[dict[str, Any]], str]:
        try:
            response = self.session.post(
                f"{self.base_url}/md",
                json={"url": self.link, "f": self.content_filter},
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            content = self._extract_markdown(data)
            if self.max_content_chars > 0:
                content = content[: self.max_content_chars]

            title = (
                data.get("title")
                or data.get("metadata", {}).get("title")
                or self._title_from_markdown(content)
            )
            return content, [], title
        except Exception as exc:
            print(f"Crawl4AI scrape error for {self.link}: {exc}")
            return "", [], ""

    async def scrape_async(self) -> tuple[str, list[dict[str, Any]], str]:
        return self.scrape()

    def _extract_markdown(self, result: Any) -> str:
        if isinstance(result, dict):
            markdown = result.get("markdown")
            if isinstance(markdown, str):
                return markdown
            if isinstance(markdown, dict):
                return (
                    markdown.get("fit_markdown")
                    or markdown.get("raw_markdown")
                    or markdown.get("markdown_with_citations")
                    or ""
                )
            return (
                result.get("fit_markdown")
                or result.get("cleaned_html")
                or result.get("html")
                or ""
            )

        markdown = getattr(result, "markdown", "")
        for candidate in (
            getattr(markdown, "fit_markdown", None),
            getattr(markdown, "raw_markdown", None),
            markdown,
            getattr(result, "fit_markdown", None),
            getattr(result, "raw_markdown", None),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate

        return ""

    @staticmethod
    def _title_from_markdown(content: str) -> str:
        match = re.search(r"^#\s+(.+)$", content or "", re.MULTILINE)
        return match.group(1).strip() if match else ""
