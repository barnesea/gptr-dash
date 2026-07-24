import json
from concurrent.futures import ThreadPoolExecutor

from gpt_researcher.utils.research_trajectory import (
    ResearchTrajectory,
    load_trajectory,
)
from evals.trajectory_compare import summarize


def test_job_scoped_trajectory_is_append_only_and_concurrency_safe(tmp_path):
    trajectory = ResearchTrajectory(
        query="test query",
        directory=tmp_path,
        job_id="job/test",
        metadata={"tree_policy": "ranked"},
    )

    def write(index):
        trajectory.record(
            "branch",
            {"index": index},
            node_id=f"root.{index}",
            parent_node_id="root",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(20)))
    trajectory.finalize("success", {"source_count": 3})

    events = load_trajectory(trajectory.path)
    assert trajectory.job_id == "job-test"
    assert events[0]["type"] == "job_started"
    assert events[-1]["type"] == "job_completed"
    assert sum(event["type"] == "branch" for event in events) == 20
    assert events[-1]["data"]["status"] == "success"
    # Every line is independently valid JSON, which permits streaming review.
    for line in trajectory.path.read_text().splitlines():
        assert json.loads(line)["job_id"] == "job-test"


def test_trajectory_summary_includes_v2_retrieval_metrics(tmp_path):
    path = tmp_path / "v2.jsonl"
    events = [
        {"job_id": "v2", "type": "job_started", "data": {}},
        {
            "job_id": "v2",
            "type": "adaptive_search_decision",
            "data": {"search_needed": False},
        },
        {
            "job_id": "v2",
            "type": "search_results",
            "data": {"lexical_collision": True},
        },
        {
            "job_id": "v2",
            "type": "canonical_source_recovery",
            "data": {
                "attempted_urls": ["https://raw.example/readme"],
                "recovered": [{"canonical_url": "https://example/repo"}],
            },
        },
        {
            "job_id": "v2",
            "type": "source_evidence_judgment",
            "data": {
                "source_count": 2,
                "accepted_count": 1,
                "rejected_count": 1,
                "fallback_used": False,
            },
        },
        {
            "job_id": "v2",
            "type": "candidate_ledger",
            "data": {"candidate_count": 4, "candidates": []},
        },
        {
            "job_id": "v2",
            "type": "job_completed",
            "data": {"status": "success"},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    summary = summarize(path)

    assert summary["candidate_count"] == 4
    assert summary["preliminary_reuse_rate"] == 1.0
    assert summary["lexical_collision_count"] == 1
    assert summary["resolver_recovery_count"] == 1
    assert summary["evidence_judgment_acceptance_rate"] == 0.5
