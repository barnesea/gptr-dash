from gpt_researcher.actions.source_selection import (
    canonical_source_alternatives,
    deterministic_select_sources,
    should_use_model_selector,
    source_quality_tier,
    has_meaningful_query_anchor,
    parse_model_selection,
    source_url,
)


def test_official_sentence_transformers_and_azure_hosts_are_primary():
    assert (
        source_quality_tier(
            {"url": "https://sbert.net/docs/package_reference/losses.html"}
        )
        == "primary"
    )
    assert (
        source_quality_tier(
            {"url": "https://ai.azure.com/catalog/models/embedding"}
        )
        == "primary"
    )


def test_canonical_source_alternatives_are_same_source_and_bounded():
    assert canonical_source_alternatives(
        "https://github.com/org/repo/blob/main/README.md"
    ) == ["https://raw.githubusercontent.com/org/repo/main/README.md"]
    assert canonical_source_alternatives(
        "https://huggingface.co/org/model/blob/main/config.json"
    ) == ["https://huggingface.co/org/model/resolve/main/config.json"]
    assert canonical_source_alternatives("https://arxiv.org/abs/2501.12345") == [
        "https://arxiv.org/html/2501.12345",
        "https://arxiv.org/pdf/2501.12345",
    ]


def test_deep_source_standards_prefer_primary_and_only_use_fallback_when_needed():
    candidates = [
        {"href": "https://docs.example.test/guide", "title": "Rust async guide", "body": "Rust async runtime guide"},
        {"href": "https://blog.example.test/async", "title": "Rust async guide", "body": "Rust async runtime guide"},
    ]
    selected, reasons = deterministic_select_sources("Rust async runtime guide", candidates, 3, strict=True)
    assert [source_quality_tier(item) for item in selected] == ["primary"]
    assert "fallback is unnecessary" in reasons[candidates[1]["href"]]


def test_deep_source_standards_reject_generic_publishing_platforms_even_without_primary_sources():
    candidates = [
        {"href": "https://medium.com/example/rust-async", "title": "Rust async runtime guide", "body": "Rust async runtime guide"},
        {"href": "https://project.example.test/guide", "title": "Rust async runtime guide", "body": "Rust async runtime guide"},
    ]
    selected, reasons = deterministic_select_sources("Rust async runtime guide", candidates, 3, strict=True)
    assert [source_url(item) for item in selected] == ["https://project.example.test/guide"]
    assert "low-value" in reasons["https://medium.com/example/rust-async"]


def test_selector_auto_skips_clear_primary_and_uses_llm_for_ambiguous_candidates():
    primary = [{"href": "https://docs.example.test/guide", "title": "Rust async guide", "body": "Rust async runtime guide"}]
    ambiguous = [
        {"href": "https://one.example.test/guide", "title": "Rust async guide", "body": "Rust async runtime guide"},
        {"href": "https://two.example.test/guide", "title": "Rust async guide", "body": "Rust async runtime guide"},
    ]
    assert not should_use_model_selector("Rust async runtime guide", primary, "auto")
    assert should_use_model_selector("Rust async runtime guide", ambiguous, "auto")


def test_general_science_source_tiers_and_social_rejection():
    assert source_quality_tier(
        {"url": "https://www.fisheries.noaa.gov/species/green-turtle"}
    ) == "primary"
    assert source_quality_tier(
        {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12345/"}
    ) == "primary"
    assert source_quality_tier(
        {"url": "https://doi.org/10.1000/example"}
    ) == "primary"
    assert source_quality_tier(
        {"url": "https://www.si.edu/spotlight/sea-turtles"}
    ) == "reputable"
    assert source_quality_tier(
        {"url": "https://www.researchgate.net/publication/12345"}
    ) == "fallback"
    assert source_quality_tier(
        {"url": "https://www.facebook.com/example/posts/123"}
    ) == "reject"
    assert source_quality_tier(
        {"url": "https://www.facebook.com/example/posts/123"},
        "Research Facebook content moderation policy",
    ) == "primary"


def test_selector_auto_skips_llm_for_decisive_fallback_score_gap():
    query = "Rust async runtime performance scheduler"
    candidates = [
        {
            "href": "https://one.example.test/guide",
            "title": "Rust async runtime performance scheduler",
            "body": "Detailed Rust async runtime performance scheduler evidence",
        },
        {
            "href": "https://two.example.test/guide",
            "title": "Rust async introduction",
            "body": "Rust async overview",
        },
    ]
    assert not should_use_model_selector(query, candidates, "auto")
from gpt_researcher.prompts import PromptFamily


def _candidate(url, title, body):
    return {"href": url, "title": title, "body": body}


def test_deterministic_selection_prefers_relevant_primary_and_diverse_sources():
    candidates = [
        _candidate("https://dictionary.example/rust", "Rust definition", "definition of rust"),
        _candidate("https://doc.rust-lang.org/book/async.html", "Async Programming in Rust", "Rust async runtime and futures documentation"),
        _candidate("https://tokio.rs/tokio/tutorial", "Tokio tutorial", "Runtime scheduling and async tasks"),
        _candidate("https://independent.example/rust-async", "Rust async runtime tradeoffs", "Independent benchmark and runtime comparison"),
        _candidate("https://doc.rust-lang.org/std/async.html", "More Rust docs", "Async reference"),
    ]

    selected, reasons = deterministic_select_sources("Rust async runtime tradeoffs", candidates, 3)
    urls = [source_url(candidate) for candidate in selected]

    assert "https://doc.rust-lang.org/book/async.html" in urls
    assert "https://independent.example/rust-async" in urls
    assert "https://dictionary.example/rust" not in urls
    assert len(urls) <= 3
    assert "low-value" in reasons["https://dictionary.example/rust"]


def test_invalid_model_selection_uses_deterministic_fallback_contract():
    candidates = [
        _candidate("https://docs.example/a", "A", "relevant query anchors"),
        _candidate("https://other.example/b", "B", "relevant query anchors"),
    ]
    assert parse_model_selection({"selected": [{"id": "bogus"}]}, candidates, 3) is None

    selected, _ = deterministic_select_sources("relevant query anchors", candidates, 1)
    assert len(selected) == 1


def test_query_anchor_guard_rejects_an_off_topic_result_card():
    assert not has_meaningful_query_anchor(
        "Linux 6.12 io_uring application changes",
        _candidate("https://blog.example/gnome", "GNOME fractional scaling", "Wayland display settings"),
    )


def test_query_anchor_guard_requires_a_specific_or_multiple_anchors():
    query = "What changed in Linux kernel 6.12 io_uring support for application developers?"
    assert not has_meaningful_query_anchor(
        query,
        _candidate("https://blog.example/gnome", "GNOME on Linux", "Linux desktop fractional scaling"),
    )
    assert has_meaningful_query_anchor(
        query,
        _candidate("https://kernel.org/6.12", "Linux 6.12 io_uring changes", "Kernel release notes"),
    )


def test_evidence_grounded_planning_prompt_uses_result_cards_and_bans_operators():
    prompt = PromptFamily.generate_search_queries_prompt(
        "Which Linux kernel changes affect io_uring in version 6.12?",
        "",
        "research_report",
        max_iterations=2,
        evidence_enabled=True,
        context=[{
            "title": "Linux 6.12 io_uring updates",
            "href": "https://kernel.org/releases/6.12",
            "body": "Kernel release notes mention io_uring changes.",
            "engine": "google",
            "date": "2026-01-03",
        }],
    )

    assert "Linux 6.12 io_uring updates" in prompt
    assert "kernel.org/releases/6.12" in prompt
    assert "Do not invent specifics" in prompt
    assert "site:" in prompt
