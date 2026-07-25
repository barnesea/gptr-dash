from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpt_researcher.skills.writer import ReportGenerator


@pytest.mark.asyncio
async def test_writer_abstains_when_ledger_has_no_verified_sources():
    cfg = SimpleNamespace(agent_role=None)
    researcher = SimpleNamespace(
        query="What are turtle predators?",
        cfg=cfg,
        role="researcher",
        report_type="deep",
        report_source="web",
        tone=None,
        websocket=None,
        headers={},
        context="coverage ledger metadata only",
        coverage_ledger=[
            {
                "aspect_id": "aspect-1",
                "state": "compression_empty",
                "verified_urls": [],
            }
        ],
        verbose=False,
        get_research_images=lambda: [],
        get_research_sources=lambda: [],
    )

    report = await ReportGenerator(researcher).write_report()

    assert report.startswith("# Evidence limitation")
    assert "no factual report was generated" in report.lower()
    assert "aspect-1" in report


@pytest.mark.asyncio
async def test_writer_can_report_labeled_supplemental_evidence_without_fact_claim():
    url = "https://practitioner.example.test/field-note"
    cfg = SimpleNamespace(agent_role=None)
    researcher = SimpleNamespace(
        query="How is the named system configured?",
        cfg=cfg,
        role="researcher",
        report_type="deep",
        report_source="web",
        tone=None,
        websocket=None,
        headers={},
        context="coverage ledger metadata and labeled evidence",
        coverage_ledger=[
            {
                "aspect_id": "aspect-1",
                "state": "scope_missing",
                "verified_urls": [],
                "evidence_pool_sources": [
                    {
                        "url": url,
                        "claim_status": "provisional",
                        "evidence_strength": "tentative",
                        "confidence_label": "medium",
                        "applicability": "partial",
                        "supported_claims": [
                            "One practitioner reports a configuration used "
                            "for one deployment mode."
                        ],
                    }
                ],
            }
        ],
        verbose=False,
        kwargs={},
        add_costs=lambda _cost: None,
        get_research_images=lambda: [],
        get_research_sources=lambda: [],
    )
    generated = (
        "A practitioner provisionally reports using this configuration for "
        "one deployment mode; this is tentative and only partially applicable "
        f"evidence. ([field note]({url}))"
    )

    with patch(
        "gpt_researcher.skills.writer.generate_report",
        new=AsyncMock(return_value=generated),
    ) as report_call:
        report = await ReportGenerator(researcher).write_report()

    assert report == generated + f"\n\n## References\n\n- [{url}]({url})\n"
    assert report_call.await_args.kwargs["verified_source_urls"] == []
    assert report_call.await_args.kwargs["supplemental_source_urls"] == [
        url
    ]


@pytest.mark.asyncio
async def test_writer_labels_supplemental_evidence_when_correction_does_not():
    url = "https://practitioner.example.test/field-note"
    cfg = SimpleNamespace(agent_role=None)
    researcher = SimpleNamespace(
        query="How is the named system configured?",
        cfg=cfg,
        role="researcher",
        report_type="deep",
        report_source="web",
        tone=None,
        websocket=None,
        headers={},
        context="coverage ledger metadata and labeled evidence",
        coverage_ledger=[
            {
                "aspect_id": "aspect-1",
                "state": "scope_missing",
                "verified_urls": [],
                "evidence_pool_sources": [
                    {
                        "url": url,
                        "claim_status": "provisional",
                        "evidence_role": "practitioner",
                        "evidence_strength": "tentative",
                        "confidence_label": "medium",
                        "applicability": "partial",
                    }
                ],
            }
        ],
        verbose=False,
        kwargs={},
        add_costs=lambda _cost: None,
        get_research_images=lambda: [],
        get_research_sources=lambda: [],
    )
    unqualified = (
        "The system always requires forty examples in production. "
        f"([field note]({url}))"
    )

    with (
        patch(
            "gpt_researcher.skills.writer.generate_report",
            new=AsyncMock(return_value=unqualified),
        ),
        patch(
            "gpt_researcher.skills.writer.repair_report_evidence_safety",
            new=AsyncMock(return_value=unqualified),
        ),
    ):
        report = await ReportGenerator(researcher).write_report()

    assert (
        "*Evidence label: provisional practitioner evidence; tentative; "
        "medium confidence; partial applicability*" in report
    )
    assert not report.startswith("> **Evidence limitation:**")
