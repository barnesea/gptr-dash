"""Per-job, append-only trajectory recording for research evaluation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid


logger = logging.getLogger(__name__)
SAFE_ID = re.compile(r"[^a-zA-Z0-9_.-]+")


class ResearchTrajectory:
    """Write one concurrency-safe JSONL event stream per research job.

    The recorder is deliberately instance-scoped.  Deep-research children
    receive the same object, avoiding the cross-job state and overwritten JSON
    files caused by the legacy process-global handler.
    """

    def __init__(
        self,
        *,
        query: str,
        directory: str | Path,
        job_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raw_id = job_id or str(uuid.uuid4())
        self.job_id = SAFE_ID.sub("-", raw_id).strip(".-") or str(uuid.uuid4())
        self.directory = Path(directory).expanduser()
        self.path = self.directory / f"{self.job_id}.jsonl"
        self.started_at = time.perf_counter()
        self._lock = threading.Lock()
        self._event_counts: Counter[str] = Counter()
        self._closed = False

        self.directory.mkdir(parents=True, exist_ok=True)
        self.record(
            "job_started",
            {
                "query": query,
                "metadata": metadata or {},
            },
            node_id="root",
        )

    def record(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        node_id: str = "root",
        parent_node_id: str | None = None,
    ) -> None:
        """Append one event. Recording failures never fail the research job."""
        if self._closed:
            return
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - self.started_at) * 1000, 1),
            "job_id": self.job_id,
            "node_id": node_id,
            "parent_node_id": parent_node_id,
            "type": event_type,
            "data": data or {},
        }
        try:
            line = json.dumps(event, ensure_ascii=False, default=str)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")
                self._event_counts[event_type] += 1
        except (OSError, TypeError, ValueError) as error:
            logger.warning("Unable to append research trajectory event: %s", error)

    def finalize(self, status: str, data: dict[str, Any] | None = None) -> None:
        """Write the terminal event once and close the recorder."""
        if self._closed:
            return
        summary = {
            "status": status,
            "duration_ms": round((time.perf_counter() - self.started_at) * 1000, 1),
            "event_counts": dict(self._event_counts),
            **(data or {}),
        }
        self.record("job_completed", summary, node_id="root")
        self._closed = True


def load_trajectory(path: str | Path) -> list[dict[str, Any]]:
    """Load a trajectory for tests and offline comparison tooling."""
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events
