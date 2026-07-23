from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import format_sources_for_response, source_urls_from_sources


def test_verified_source_urls_exclude_attempted_or_rejected_urls():
    sources = [
        {"title": "A", "url": "https://example.com/a", "raw_content": "alpha"},
        {"title": "A duplicate", "href": "https://example.com/a", "body": "duplicate"},
        {"title": "B", "href": "https://example.org/b", "content": "bravo"},
        {"title": "Missing URL", "raw_content": "ignored"},
    ]

    assert source_urls_from_sources(sources) == [
        "https://example.com/a",
        "https://example.org/b",
    ]


def test_source_response_counts_scraped_content_fields():
    formatted = format_sources_for_response(
        [{"title": "Evidence", "href": "https://example.com", "raw_content": "12345"}]
    )

    assert formatted == [
        {
            "title": "Evidence",
            "url": "https://example.com",
            "content_length": 5,
        }
    ]
