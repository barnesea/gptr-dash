import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpt_researcher.utils.research_budget import (
    build_research_policy,
    cold_start_policy,
    execution_policy,
    research_stack_fingerprint,
    validate_research_duration,
)


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (15, (1, 0, 0, 8, 2)),
        (29, (1, 0, 0, 8, 2)),
        (30, (2, 1, 0, 8, 2)),
        (59, (2, 1, 0, 8, 2)),
        (60, (3, 2, 1, 10, 3)),
        (119, (3, 2, 1, 10, 3)),
        (120, (4, 3, 2, 10, 3)),
        (239, (4, 3, 2, 10, 3)),
        (240, (5, 5, 3, 12, 3)),
        (419, (5, 5, 3, 12, 3)),
        (420, (6, 8, 4, 15, 3)),
        (600, (6, 8, 4, 15, 3)),
    ],
)
def test_cold_start_policy_bands(duration, expected):
    policy = cold_start_policy(duration)
    assert (
        policy.aspect_count,
        policy.repair_allowance,
        policy.max_deepened_branches,
        policy.result_cards_per_query,
        policy.scrape_cap_per_query,
    ) == expected
    assert policy.max_depth == 2
    assert policy.concurrency_limit == 4


@pytest.mark.parametrize("value", [14, 601, True, 30.5, "bad"])
def test_duration_validation_rejects_out_of_range_or_non_integer(value):
    with pytest.raises(ValueError):
        validate_research_duration(value)


def make_cfg(tmp_path: Path, *, model: str = "model-a", mode: str = "enabled"):
    return SimpleNamespace(
        fast_llm=model,
        smart_llm=model,
        strategic_llm=model,
        embedding="openai:jina-v5-retrieval",
        retrievers=["searx"],
        scraper="crawl4ai",
        deep_research_concurrency=4,
        deep_research_breadth=3,
        deep_research_depth=2,
        deep_research_max_deepened_branches=1,
        max_search_results_per_query=5,
        pre_scrape_max_sources_per_query=3,
        research_duration_controller_mode=mode,
        research_budget_calibration_enabled=True,
        research_budget_calibration_min_samples=10,
        research_trajectory_dir=str(tmp_path),
    )


def test_shadow_executes_legacy_policy_but_retains_calculated_policy(tmp_path):
    cfg = make_cfg(tmp_path, mode="shadow")
    calculated = build_research_policy(300, cfg)
    active = execution_policy(calculated, cfg)

    assert calculated.aspect_count == 5
    assert calculated.repair_allowance == 5
    assert active.aspect_count == 3
    assert active.repair_allowance == 0
    assert active.max_deepened_branches == 1


def test_calibration_uses_matching_stack_after_ten_successes(tmp_path):
    cfg = make_cfg(tmp_path)
    fingerprint = research_stack_fingerprint(cfg)
    for index in range(10):
        path = tmp_path / f"sample-{index}.jsonl"
        events = [
            {
                "type": "research_budget",
                "data": {
                    "requested_duration_seconds": 60,
                    "calibration_fingerprint": fingerprint,
                },
            },
            {
                "type": "job_completed",
                "data": {
                    "status": "success",
                    "actual_research_duration_seconds": 100 + index,
                },
            },
        ]
        path.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )

    policy = build_research_policy(60, cfg)

    assert policy.calibration_source == "local_p75"
    assert policy.calibration_sample_count == 10
    assert policy.max_deepened_branches == 0
    assert (tmp_path / "research_budget_calibration.json").exists()


def test_calibration_expands_fast_stack_within_global_caps(tmp_path):
    cfg = make_cfg(tmp_path)
    fingerprint = research_stack_fingerprint(cfg)
    for index in range(10):
        path = tmp_path / f"fast-{index}.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "research_budget",
                            "data": {
                                "requested_duration_seconds": 120,
                                "calibration_fingerprint": fingerprint,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "job_completed",
                            "data": {
                                "status": "success",
                                "actual_research_duration_seconds": 45 + index,
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    policy = build_research_policy(120, cfg)

    assert policy.calibration_source == "local_p75"
    assert policy.aspect_count == 6
    assert policy.repair_allowance == 8
    assert policy.max_deepened_branches == 4
    assert policy.result_cards_per_query == 15
    assert policy.aspect_count <= 6
    assert policy.repair_allowance <= 8
    assert policy.max_deepened_branches <= 4


def test_corrupt_and_other_stack_samples_fall_back_to_cold_start(tmp_path):
    cfg = make_cfg(tmp_path)
    (tmp_path / "corrupt.jsonl").write_text("{broken", encoding="utf-8")
    other = make_cfg(tmp_path, model="other-model")
    other_fingerprint = research_stack_fingerprint(other)
    (tmp_path / "other.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "research_budget",
                        "data": {
                            "requested_duration_seconds": 60,
                            "calibration_fingerprint": other_fingerprint,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "job_completed",
                        "data": {
                            "status": "success",
                            "actual_research_duration_seconds": 90,
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    policy = build_research_policy(60, cfg)
    assert policy.calibration_source == "cold_start"
    assert policy.calibration_sample_count == 0
