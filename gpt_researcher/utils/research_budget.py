"""Duration-based policy and local timing calibration for deep research."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MIN_RESEARCH_DURATION_SECONDS = 15
MAX_RESEARCH_DURATION_SECONDS = 600
MAX_CALIBRATED_ASPECTS = 6
MAX_CALIBRATED_REPAIRS = 8
MAX_CALIBRATED_DEEPENED_BRANCHES = 4
POLICY_SCHEMA_VERSION = 3
CALIBRATION_FORMAT_VERSION = 3


@dataclass(frozen=True)
class ResearchPolicy:
    """All execution limits calculated once at the start of a research job."""

    requested_duration_seconds: int
    aspect_count: int
    repair_allowance: int
    max_repairs_per_aspect: int
    max_deepened_branches: int
    result_cards_per_query: int
    scrape_cap_per_query: int
    max_depth: int = 2
    concurrency_limit: int = 4
    estimated_research_seconds: float = 0.0
    estimated_stage_seconds: dict[str, float] = field(default_factory=dict)
    calibration_source: str = "cold_start"
    calibration_sample_count: int = 0
    calibration_fingerprint: str = ""
    controller_mode: str = "enabled"

    @property
    def work_units(self) -> float:
        return _policy_work_units(self)

    @property
    def policy_signature(self) -> str:
        payload = {
            "version": POLICY_SCHEMA_VERSION,
            "aspect_count": self.aspect_count,
            "repair_allowance": self.repair_allowance,
            "max_repairs_per_aspect": self.max_repairs_per_aspect,
            "max_deepened_branches": self.max_deepened_branches,
            "result_cards_per_query": self.result_cards_per_query,
            "scrape_cap_per_query": self.scrape_cap_per_query,
            "max_depth": self.max_depth,
            "concurrency_limit": self.concurrency_limit,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "policy_schema_version": POLICY_SCHEMA_VERSION,
            "policy_signature": self.policy_signature,
            "work_units": self.work_units,
        }


def validate_research_duration(value: Any) -> int:
    """Return a validated integer research duration in the public range."""
    if isinstance(value, bool):
        raise ValueError("research_duration_seconds must be an integer from 15 to 600")
    try:
        duration = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "research_duration_seconds must be an integer from 15 to 600"
        ) from exc
    if duration != value and not (
        isinstance(value, str) and value.strip() == str(duration)
    ):
        raise ValueError("research_duration_seconds must be a whole number")
    if not MIN_RESEARCH_DURATION_SECONDS <= duration <= MAX_RESEARCH_DURATION_SECONDS:
        raise ValueError("research_duration_seconds must be between 15 and 600")
    return duration


def cold_start_policy(duration: int, *, concurrency_limit: int = 4) -> ResearchPolicy:
    """Map a duration to the deliberately bounded cold-start policy table."""
    duration = validate_research_duration(duration)
    if duration < 30:
        values = (1, 0, 0, 0, 8, 2)
    elif duration < 60:
        values = (2, 1, 1, 0, 8, 2)
    elif duration < 120:
        values = (3, 2, 2, 1, 10, 3)
    elif duration < 240:
        values = (4, 3, 2, 2, 10, 3)
    elif duration < 420:
        values = (5, 5, 2, 3, 12, 3)
    else:
        values = (6, 8, 2, 4, 15, 3)
    (
        aspect_count,
        repair_allowance,
        max_repairs_per_aspect,
        max_deepened_branches,
        result_cards,
        scrape_cap,
    ) = values
    return ResearchPolicy(
        requested_duration_seconds=duration,
        aspect_count=aspect_count,
        repair_allowance=repair_allowance,
        max_repairs_per_aspect=max_repairs_per_aspect,
        max_deepened_branches=max_deepened_branches,
        result_cards_per_query=result_cards,
        scrape_cap_per_query=scrape_cap,
        concurrency_limit=max(1, min(int(concurrency_limit), 4)),
        estimated_research_seconds=float(duration),
        estimated_stage_seconds={
            "planning": round(duration * 0.12, 3),
            "search": round(duration * 0.12, 3),
            "selection": round(duration * 0.03, 3),
            "scraping": round(duration * 0.23, 3),
            "resolution": round(duration * 0.05, 3),
            "compression": round(duration * 0.12, 3),
            "evidence_judgment": round(duration * 0.12, 3),
            "repairs": round(duration * (0.10 if repair_allowance else 0.0), 3),
            "deepening": round(
                duration * (0.11 if max_deepened_branches else 0.0), 3
            ),
            "reporting": 0.0,
        },
    )


def research_stack_fingerprint(cfg: Any) -> str:
    """Key calibration samples to the model and retrieval stack that produced them."""
    payload = {
        "fast_llm": getattr(cfg, "fast_llm", ""),
        "smart_llm": getattr(cfg, "smart_llm", ""),
        "strategic_llm": getattr(cfg, "strategic_llm", ""),
        "embedding": getattr(cfg, "embedding", ""),
        "embedding_base_url": os.getenv("EMBEDDING_OPENAI_BASE_URL", ""),
        "retrievers": sorted(getattr(cfg, "retrievers", []) or []),
        "scraper": getattr(cfg, "scraper", ""),
        "concurrency": int(getattr(cfg, "deep_research_concurrency", 4)),
        "source_selector_mode": getattr(cfg, "source_selector_mode", "auto"),
        "retrieval_pipeline_mode": getattr(
            cfg, "retrieval_pipeline_mode", "legacy"
        ),
        "source_evidence_judge_mode": getattr(
            cfg, "source_evidence_judge_mode", "all"
        ),
        "source_evidence_judge_fallback": getattr(
            cfg, "source_evidence_judge_fallback", "hybrid"
        ),
        "canonical_content_resolution": bool(
            getattr(cfg, "canonical_content_resolution", True)
        ),
        "pdf_connect_timeout": float(
            getattr(cfg, "deep_research_pdf_connect_timeout_seconds", 3.0)
        ),
        "pdf_total_timeout": float(
            getattr(cfg, "deep_research_pdf_total_timeout_seconds", 8.0)
        ),
        "policy_schema_version": POLICY_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _percentile_75(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values if float(value) > 0)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(len(ordered) * 0.75) - 1)]


def _read_successful_samples(
    directory: Path,
    *,
    fingerprint: str,
    duration_band: tuple[int, int],
    limit: int = 50,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not directory.exists():
        return samples
    files = sorted(directory.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in files:
        budget: dict[str, Any] | None = None
        completed: dict[str, Any] | None = None
        critical_path: dict[str, Any] | None = None
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    entry = json.loads(line)
                    if entry.get("type") == "research_budget":
                        budget = entry.get("data") or {}
                    elif entry.get("type") == "critical_path_timing":
                        critical_path = entry.get("data") or {}
                    elif entry.get("type") == "job_completed":
                        completed = entry.get("data") or {}
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if not budget or not completed or completed.get("status") != "success":
            continue
        calculated = budget.get("calculated_policy") or budget
        executed = budget.get("execution_policy")
        if not isinstance(calculated, dict) or not isinstance(executed, dict):
            # Version-one trajectories did not prove which policy ran.
            continue
        if str(calculated.get("controller_mode") or "").lower() != "enabled":
            continue
        if (
            calculated.get("policy_signature")
            != executed.get("policy_signature")
        ):
            continue
        if int(calculated.get("policy_schema_version") or 0) != POLICY_SCHEMA_VERSION:
            continue
        requested = int(calculated.get("requested_duration_seconds") or 0)
        if calculated.get("calibration_fingerprint") != fingerprint:
            continue
        if not duration_band[0] <= requested <= duration_band[1]:
            continue
        actual = completed.get("actual_research_duration_seconds")
        planning = float(
            (critical_path or {}).get("planning_wall_seconds")
            or (critical_path or {}).get("planning_seconds")
            or 0
        )
        research_tree = float(
            (critical_path or {}).get("research_tree_wall_seconds")
            or (critical_path or {}).get("research_tree_seconds")
            or 0
        )
        work_units = float(
            (critical_path or {}).get("executed_work_units")
            or executed.get("work_units")
            or 0
        )
        if (
            isinstance(actual, (int, float))
            and actual > 0
            and planning >= 0
            and research_tree > 0
            and work_units > 0
        ):
            samples.append(
                {
                    "actual_research_seconds": float(actual),
                    "planning_seconds": planning,
                    "research_tree_seconds": research_tree,
                    "tree_seconds_per_work_unit": research_tree / work_units,
                    "work_units": work_units,
                }
            )
        if len(samples) >= limit:
            break
    return samples


def _duration_band(duration: int) -> tuple[int, int]:
    if duration < 30:
        return 15, 29
    if duration < 60:
        return 30, 59
    if duration < 120:
        return 60, 119
    if duration < 240:
        return 120, 239
    if duration < 420:
        return 240, 419
    return 420, 600


def _persist_calibration(
    directory: Path,
    *,
    fingerprint: str,
    sample_count: int,
    p75_research_seconds: float,
    p75_planning_seconds: float,
    p75_tree_seconds_per_work_unit: float,
) -> None:
    """Persist a compact, non-secret calibration snapshot atomically."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "research_budget_calibration.json"
    payload: dict[str, Any] = {
        "version": CALIBRATION_FORMAT_VERSION,
        "stacks": {},
    }
    try:
        if destination.exists():
            loaded = json.loads(destination.read_text(encoding="utf-8"))
            if (
                isinstance(loaded, dict)
                and loaded.get("version") == CALIBRATION_FORMAT_VERSION
                and isinstance(loaded.get("stacks"), dict)
            ):
                payload = loaded
    except (OSError, json.JSONDecodeError):
        pass
    payload["stacks"][fingerprint] = {
        "sample_count": sample_count,
        "p75_research_seconds": round(p75_research_seconds, 3),
        "p75_planning_seconds": round(p75_planning_seconds, 3),
        "p75_tree_seconds_per_work_unit": round(
            p75_tree_seconds_per_work_unit, 3
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, delete=False
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_name = handle.name
        os.replace(temp_name, destination)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _policy_work_units(policy: ResearchPolicy) -> float:
    """Approximate conditional work without treating repair slots as full jobs."""
    return (
        float(policy.aspect_count)
        + 0.5 * float(policy.repair_allowance)
        + 2.0 * float(policy.max_deepened_branches)
    )


def _cards_for_aspects(aspect_count: int) -> int:
    if aspect_count <= 2:
        return 8
    if aspect_count <= 4:
        return 10
    if aspect_count == 5:
        return 12
    return 15


def _calibrate_policy(
    policy: ResearchPolicy,
    planning_p75: float,
    tree_seconds_per_work_unit_p75: float,
) -> ResearchPolicy:
    """Scale bounded work down or up using matching-stack p75 observations."""
    if tree_seconds_per_work_unit_p75 <= 0:
        return policy
    calibrated = policy

    def estimate(candidate: ResearchPolicy) -> float:
        return planning_p75 + (
            tree_seconds_per_work_unit_p75 * _policy_work_units(candidate)
        )

    estimated = estimate(calibrated)
    while estimated > policy.requested_duration_seconds * 1.05:
        if calibrated.max_deepened_branches > 0:
            calibrated = replace(
                calibrated,
                max_deepened_branches=calibrated.max_deepened_branches - 1,
            )
            estimated = estimate(calibrated)
            continue
        if calibrated.repair_allowance > 0:
            calibrated = replace(
                calibrated,
                repair_allowance=calibrated.repair_allowance - 1,
            )
            estimated = estimate(calibrated)
            continue
        if calibrated.aspect_count > 1:
            aspect_count = calibrated.aspect_count - 1
            calibrated = replace(
                calibrated,
                aspect_count=aspect_count,
                result_cards_per_query=min(
                    calibrated.result_cards_per_query,
                    _cards_for_aspects(aspect_count),
                ),
            )
            estimated = estimate(calibrated)
            continue
        break

    # A fast local stack should spend a longer requested budget on additional
    # coverage and selective recovery/deepening, never on sleeping. Expansion
    # remains inside the public global caps and only begins after the minimum
    # matching-stack sample count is met.
    while estimated < policy.requested_duration_seconds * 0.75:
        if calibrated.aspect_count < MAX_CALIBRATED_ASPECTS:
            aspect_count = calibrated.aspect_count + 1
            calibrated = replace(
                calibrated,
                aspect_count=aspect_count,
                result_cards_per_query=max(
                    calibrated.result_cards_per_query,
                    _cards_for_aspects(aspect_count),
                ),
                scrape_cap_per_query=max(
                    calibrated.scrape_cap_per_query,
                    3 if aspect_count >= 3 else 2,
                ),
            )
        elif calibrated.repair_allowance < MAX_CALIBRATED_REPAIRS:
            calibrated = replace(
                calibrated,
                repair_allowance=calibrated.repair_allowance + 1,
            )
        elif calibrated.max_deepened_branches < min(
            MAX_CALIBRATED_DEEPENED_BRANCHES,
            calibrated.aspect_count,
        ):
            calibrated = replace(
                calibrated,
                max_deepened_branches=calibrated.max_deepened_branches + 1,
            )
        else:
            break
        estimated = estimate(calibrated)
    return replace(calibrated, estimated_research_seconds=round(estimated, 3))


def build_research_policy(duration: Any, cfg: Any) -> ResearchPolicy:
    """Build the complete immutable policy before the job begins."""
    requested = validate_research_duration(duration)
    mode = str(getattr(cfg, "research_duration_controller_mode", "off")).lower()
    if mode not in {"off", "shadow", "enabled"}:
        mode = "off"
    fingerprint = research_stack_fingerprint(cfg)
    policy = replace(
        cold_start_policy(
            requested,
            concurrency_limit=getattr(cfg, "deep_research_concurrency", 4),
        ),
        calibration_fingerprint=fingerprint,
        controller_mode=mode,
    )
    if not getattr(cfg, "research_budget_calibration_enabled", True):
        return policy

    directory = Path(getattr(cfg, "research_trajectory_dir", "data/trajectories"))
    samples = _read_successful_samples(
        directory,
        fingerprint=fingerprint,
        duration_band=_duration_band(requested),
        limit=50,
    )
    p75 = _percentile_75(sample["actual_research_seconds"] for sample in samples)
    p75_planning = _percentile_75(
        sample["planning_seconds"] for sample in samples
    )
    p75_tree_per_unit = _percentile_75(
        sample["tree_seconds_per_work_unit"] for sample in samples
    )
    try:
        _persist_calibration(
            directory,
            fingerprint=fingerprint,
            sample_count=len(samples),
            p75_research_seconds=p75,
            p75_planning_seconds=p75_planning,
            p75_tree_seconds_per_work_unit=p75_tree_per_unit,
        )
    except OSError:
        pass

    minimum = max(1, int(getattr(cfg, "research_budget_calibration_min_samples", 10)))
    if len(samples) < minimum:
        return replace(policy, calibration_sample_count=len(samples))
    calibrated = _calibrate_policy(
        policy, p75_planning, p75_tree_per_unit
    )
    return replace(
        calibrated,
        estimated_stage_seconds={
            **calibrated.estimated_stage_seconds,
            "planning": round(p75_planning, 3),
            "research_tree": round(
                p75_tree_per_unit * calibrated.work_units, 3
            ),
        },
        calibration_source="local_p75",
        calibration_sample_count=len(samples),
    )


def execution_policy(policy: ResearchPolicy, cfg: Any) -> ResearchPolicy:
    """Return the policy to execute, preserving 2/3 focused rollback in off/shadow."""
    if policy.controller_mode == "enabled":
        return policy
    return replace(
        policy,
        aspect_count=max(1, int(getattr(cfg, "deep_research_breadth", 3))),
        repair_allowance=0,
        max_repairs_per_aspect=0,
        max_deepened_branches=max(
            0, int(getattr(cfg, "deep_research_max_deepened_branches", 1))
        ),
        result_cards_per_query=max(
            1, int(getattr(cfg, "max_search_results_per_query", 5))
        ),
        scrape_cap_per_query=max(
            1, int(getattr(cfg, "pre_scrape_max_sources_per_query", 3))
        ),
        max_depth=max(1, min(int(getattr(cfg, "deep_research_depth", 2)), 2)),
    )


def report_word_target(duration: Any) -> int:
    """Return a report-size target proportional to the research request."""
    seconds = validate_research_duration(duration)
    if seconds < 30:
        return 300
    if seconds < 60:
        return 500
    if seconds < 120:
        return 800
    if seconds < 240:
        return 1200
    if seconds < 420:
        return 1600
    return 2000
