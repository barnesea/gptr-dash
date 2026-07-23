"""Adversarial source pools based on difficult research questions."""

import pytest

from gpt_researcher.actions.source_selection import deterministic_select_sources, source_url


def card(url, title, snippet):
    return {"href": url, "title": title, "body": snippet}


@pytest.mark.parametrize(
    ("query", "candidates", "must_keep", "must_reject", "reason_fragment"),
    [
        (
            "EU Cyber Resilience Act open-source software steward obligations exemptions phased deadlines",
            [
                card("https://digital-strategy.ec.europa.eu/en/policies/cyber-resilience-act", "Cyber Resilience Act", "European Commission overview of obligations and dates"),
                card("https://eur-lex.europa.eu/eli/reg/2024/2847/oj", "Regulation (EU) 2024/2847", "Official text with open-source steward provisions"),
                card("https://www.eclipse.org/legal/cra-guide", "CRA guide for open source", "Independent maintainer guidance and implementation analysis"),
                card("https://dictionary.example/steward", "Steward definition", "Definition and synonyms"),
                card("https://text-compare.example/eu-cra", "Compare texts", "Utility for text comparison"),
            ],
            "https://eur-lex.europa.eu/eli/reg/2024/2847/oj",
            "https://dictionary.example/steward",
            "low-value",
        ),
        (
            "OpenTelemetry semantic conventions 1.30 HTTP server attributes changes from 1.29",
            [
                card("https://opentelemetry.io/docs/specs/semconv/http/", "HTTP semantic conventions", "Official OpenTelemetry HTTP server attribute specification"),
                card("https://github.com/open-telemetry/semantic-conventions/releases/tag/v1.30.0", "v1.30.0 release", "Release notes for semantic convention changes"),
                card("https://honeycomb.io/blog/otel-semconv-1-30", "Migration notes", "Independent practical migration coverage"),
                card("https://dictionary.example/telemetry", "Telemetry definition", "Dictionary definition"),
            ],
            "https://github.com/open-telemetry/semantic-conventions/releases/tag/v1.30.0",
            "https://dictionary.example/telemetry",
            "low-value",
        ),
        (
            "Linux kernel 6.12 io_uring async discard changes for application developers",
            [
                card("https://kernel.org/doc/html/latest/io_uring/", "io_uring documentation", "Official Linux kernel io_uring documentation"),
                card("https://lwn.net/Articles/1000000/", "Linux 6.12 io_uring async discard", "Independent technical coverage of the 6.12 feature"),
                card("https://blog.example/gnome", "GNOME fractional scaling on Linux", "Linux desktop Wayland changes"),
                card("https://diffchecker.example/linux", "Linux text comparison", "Online diff utility"),
            ],
            "https://lwn.net/Articles/1000000/",
            "https://blog.example/gnome",
            "no meaningful query-anchor",
        ),
    ],
)
def test_adversarial_research_pools_prefer_evidence_and_reject_noise(query, candidates, must_keep, must_reject, reason_fragment):
    selected, reasons = deterministic_select_sources(query, candidates, max_sources=3)
    urls = [source_url(candidate) for candidate in selected]

    assert must_keep in urls
    assert must_reject not in urls
    assert len(urls) <= 3
    assert reason_fragment in reasons[must_reject]
    assert len({url.split("/")[2] for url in urls}) == len(urls)
