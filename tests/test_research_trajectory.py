import json
from concurrent.futures import ThreadPoolExecutor

from gpt_researcher.utils.research_trajectory import (
    ResearchTrajectory,
    load_trajectory,
)


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
