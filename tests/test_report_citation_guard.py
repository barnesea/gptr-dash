from gpt_researcher.actions.report_generation import (
    add_visible_evidence_limitation,
    enforce_verified_citation_urls,
    report_citation_urls,
    report_quality_diagnostics,
)


def test_report_citation_guard_rewrites_or_removes_unverified_urls():
    verified = "https://huggingface.co/microsoft/Mage-Flow"
    report = f"""# Mage-Flow

The verified model card describes the model ([model card]({verified})).
It links to an unverified sibling ([edit model](https://huggingface.co/microsoft/Mage-Flow-Edit)).
An unrelated URL must not survive ([blog](https://example.com/speculation)).

## References

- [Mage-Flow]({verified})
- [Edit](https://huggingface.co/microsoft/Mage-Flow-Edit)
- [Speculation](https://example.com/speculation)
"""

    guarded = enforce_verified_citation_urls(report, [verified])

    assert report_citation_urls(guarded) == {verified}
    assert "https://huggingface.co/microsoft/Mage-Flow-Edit" not in guarded
    assert "https://example.com/speculation" not in guarded
    assert guarded.count("## References") == 1


def test_report_citation_guard_strips_all_links_without_verified_sources():
    guarded = enforce_verified_citation_urls(
        "Claim ([source](https://unverified.example/a)).\n\n## References\n",
        [],
    )

    assert report_citation_urls(guarded) == set()
    assert "https://" not in guarded
    assert "## References" not in guarded


def test_report_quality_guard_detects_references_only_and_scope_overclaim():
    url = "https://www.fisheries.noaa.gov/species/green-turtle"
    report = f"""# Turtle predators

This is a comprehensive account of turtle predators across sea turtles,
freshwater turtles, and tortoises, even though only sea-turtle evidence was
retrieved and this substantive paragraph has no inline citation.

## References

- [NOAA]({url})
"""
    diagnostics = report_quality_diagnostics(
        report,
        [
                {
                    "aspect_id": "freshwater",
                    "state": "scope_missing",
                    "source_tiers": {"primary": 1},
                    "missing_scope_anchors": [
                        "freshwater turtles",
                        "tortoises",
                    ],
                }
        ],
        query="What are the natural predators of turtles?",
    )
    assert not diagnostics["passes"]
    assert diagnostics["references_only"]
    assert diagnostics["unsupported_comprehensive_claim"]
    assert diagnostics["unsupported_scope_claims"] == [
        "freshwater turtles",
        "tortoises",
    ]
    limited = add_visible_evidence_limitation(report, diagnostics)
    assert limited.startswith("> **Evidence limitation:**")
    assert "comprehensive" not in limited.lower()


def test_report_quality_guard_separates_disease_from_predators():
    url = "https://example.test/turtles"
    report = (
        "Fibropapillomatosis disease is a major predator of turtles and reduces "
        f"survival. ([source]({url}))"
    )
    diagnostics = report_quality_diagnostics(
        report,
        [{"aspect_id": "predators", "state": "evidence_ready"}],
        query="What are turtle predators?",
    )
    assert diagnostics["category_error"]
