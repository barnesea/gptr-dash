from gpt_researcher.actions.report_generation import (
    add_visible_evidence_limitation,
    enforce_verified_citation_urls,
    qualify_supplemental_evidence_paragraphs,
    qualify_uncited_synthesis_paragraphs,
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


def test_report_citation_guard_handles_bare_urls():
    verified = "https://research.fs.usda.gov/download/treesearch/68855.pdf"
    guarded = enforce_verified_citation_urls(
        (
            f"Supported source: {verified}\n"
            "Unsupported source: https://example.test/speculation"
        ),
        [verified],
    )

    assert "https://example.test/speculation" not in guarded
    assert f"[source]({verified})" in guarded
    assert report_citation_urls(guarded) == {verified}


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


def test_report_quality_guard_rejects_microbes_and_predator_immunity():
    url = "https://example.test/turtles"
    report = (
        "Microbes are important turtle predators. Adult turtles are immune "
        f"to predators because of their shells. ([source]({url}))"
    )
    diagnostics = report_quality_diagnostics(
        report,
        [{"aspect_id": "predators", "state": "evidence_ready"}],
        query="What are turtle predators?",
    )
    assert diagnostics["category_error"]
    assert diagnostics["predator_immunity_overclaim"]


def test_report_guard_requires_labels_for_supplemental_evidence():
    url = "https://practitioner.example.test/field-note"
    unqualified = (
        "The system always requires forty examples and this is the recommended "
        f"production configuration. ([field note]({url}))"
    )
    diagnostics = report_quality_diagnostics(
        unqualified,
        [{"aspect_id": "workflow", "state": "scope_missing"}],
        query="How should the system be configured?",
        supplemental_source_urls=[url],
    )
    assert not diagnostics["passes"]
    assert diagnostics["unlabeled_supplemental_paragraphs"]

    labeled = (
        "One practitioner provisionally reports using forty examples; this is "
        "tentative, partially applicable evidence rather than an established "
        f"requirement. ([field note]({url}))"
    )
    diagnostics = report_quality_diagnostics(
        labeled,
        [{"aspect_id": "workflow", "state": "scope_missing"}],
        query="How should the system be configured?",
        supplemental_source_urls=[url],
    )
    assert diagnostics["unlabeled_supplemental_paragraphs"] == []


def test_report_guard_deterministically_labels_supplemental_evidence():
    provisional = "https://practitioner.example.test/field-note"
    background = "https://docs.example.test/overview"
    report = f"""# Configuration

The system always requires forty examples in production.
([field note]({provisional}))

The toolkit accepts image folders with caption files.
([overview]({background}))

## References

- [field note]({provisional})
- [overview]({background})
"""
    ledger = [
        {
            "aspect_id": "workflow",
            "state": "scope_missing",
            "evidence_pool_sources": [
                {
                    "url": provisional,
                    "claim_status": "provisional",
                    "evidence_role": "practitioner",
                    "confidence_label": "medium",
                    "applicability": "partial",
                },
                {
                    "url": background,
                    "claim_status": "background",
                    "evidence_role": "first_party",
                    "confidence_label": "high",
                    "applicability": "adjacent",
                },
            ],
        }
    ]

    qualified = qualify_supplemental_evidence_paragraphs(report, ledger)

    assert (
        "*Evidence label: provisional practitioner evidence; tentative; "
        "medium confidence; partial applicability*" in qualified
    )
    assert (
        "*Evidence label: background or contextual evidence; not "
        "synthesis-ready; high confidence; adjacent applicability*"
        in qualified
    )
    diagnostics = report_quality_diagnostics(
        qualified,
        ledger,
        query="How should the system be configured?",
        supplemental_source_urls=[provisional, background],
    )
    assert diagnostics["unlabeled_supplemental_paragraphs"] == []


def test_report_guard_preserves_existing_supplemental_attribution():
    url = "https://practitioner.example.test/field-note"
    report = (
        "A practitioner provisionally reports using forty examples; this is "
        "tentative rather than an established requirement. "
        f"([field note]({url}))"
    )
    ledger = [
        {
            "evidence_pool_sources": [
                {
                    "url": url,
                    "claim_status": "provisional",
                    "evidence_role": "practitioner",
                }
            ]
        }
    ]

    assert qualify_supplemental_evidence_paragraphs(report, ledger) == report


def test_report_guard_labels_synthesis_ready_partial_applicability():
    url = "https://docs.example.test/atlasdb-cache-tuning"
    report = (
        "The guide recommends a 32 GiB cache for all AtlasDB deployments. "
        f"([guide]({url}))"
    )
    ledger = [
        {
            "aspect_id": "atlasdb-v4",
            "state": "evidence_ready",
            "evidence_pool_sources": [
                {
                    "url": url,
                    "claim_status": "synthesis_ready",
                    "evidence_role": "reputable_secondary",
                    "evidence_strength": "moderate",
                    "confidence_label": "medium",
                    "applicability": "partial",
                    "supported_scope": ["AtlasDB v3 on Linux"],
                }
            ],
        }
    ]

    before = report_quality_diagnostics(
        report,
        ledger,
        query="How should AtlasDB v4 be configured?",
    )
    assert before["unlabeled_applicability_paragraphs"]

    qualified = qualify_supplemental_evidence_paragraphs(report, ledger)

    assert qualified.endswith(
        "*Evidence label: partially applicable evidence; synthesis-ready "
        "only within its supported scope; medium confidence; partial "
        "applicability; scope: AtlasDB v3 on Linux*"
    )
    after = report_quality_diagnostics(
        qualified,
        ledger,
        query="How should AtlasDB v4 be configured?",
    )
    assert after["unlabeled_applicability_paragraphs"] == []


def test_report_guard_labels_synthesis_ready_practitioner_provenance():
    url = "https://operator.example.test/atlasdb-v4"
    report = (
        "A 48 GiB cache produced stable throughput in the tested deployment. "
        f"([operator note]({url}))"
    )
    ledger = [
        {
            "aspect_id": "atlasdb-v4",
            "state": "evidence_ready",
            "evidence_pool_sources": [
                {
                    "url": url,
                    "claim_status": "synthesis_ready",
                    "evidence_role": "practitioner",
                    "confidence_label": "high",
                    "applicability": "exact",
                    "supported_scope": ["one AtlasDB v4 deployment"],
                }
            ],
        }
    ]

    qualified = qualify_supplemental_evidence_paragraphs(report, ledger)

    assert qualified.endswith(
        "*Evidence label: corroborated practitioner evidence; "
        "synthesis-ready only within its supported scope; high confidence; "
        "exact applicability; scope: one AtlasDB v4 deployment*"
    )


def test_report_guard_labels_uncited_synthesis_locally():
    report = """# AtlasDB configuration

The recommended cache size is 32 GiB for every production deployment, and
operators should treat that value as the universal default.

## Evidence gaps

The available evidence does not establish a universal cache size.
"""

    qualified = qualify_uncited_synthesis_paragraphs(report)

    assert (
        "universal default. *Evidence label: uncited synthesis; not "
        "independently verified*" in qualified
    )
    assert qualified.count("uncited synthesis") == 1
