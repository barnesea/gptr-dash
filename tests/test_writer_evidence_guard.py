from types import SimpleNamespace

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
