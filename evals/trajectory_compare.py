"""Summarize job-scoped GPT Researcher trajectories for policy comparisons."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def summarize(path: Path) -> dict[str, Any]:
    events = load_events(path)
    counts = Counter(event.get("type", "") for event in events)
    policy: dict[str, Any] = {}
    queries: list[str] = []
    selected_urls: set[str] = set()
    rejected_urls: set[str] = set()
    source_tiers: Counter[str] = Counter()
    accepted_scrapes = 0
    rejected_scrapes = 0
    max_active = 0
    final: dict[str, Any] = {}
    budget: dict[str, Any] = {}
    coverage_entries: list[dict[str, Any]] = []
    recovery_reasons: Counter[str] = Counter()
    corroboration_candidates = 0
    corroborated = 0
    child_queries = 0
    drifted_child_queries = 0
    compression_events = 0
    compression_rescues = 0
    stage_timings: Counter[str] = Counter()

    for event in events:
        event_type = event.get("type")
        data = event.get("data") or {}
        if event_type == "deep_research_policy":
            policy = data
        elif event_type == "research_budget":
            budget = data
        elif event_type == "sub_query":
            query = str(data.get("query") or "")
            if query and query not in queries:
                queries.append(query)
        elif event_type == "source_selection":
            for selected in data.get("selected") or []:
                url = str(selected.get("url") or "")
                if url:
                    selected_urls.add(url)
                source_tiers[str(selected.get("tier") or "unknown")] += 1
            for rejected in data.get("rejected") or []:
                url = str(rejected.get("url") or "")
                if url:
                    rejected_urls.add(url)
        elif event_type == "post_scrape_source_integrity":
            accepted_scrapes += int(data.get("accepted_count") or 0)
            rejected_scrapes += int(data.get("rejected_count") or 0)
        elif event_type == "deep_research_branch":
            max_active = max(max_active, int(data.get("active_branches") or 0))
        elif event_type == "coverage_ledger":
            coverage_entries = data.get("entries") or []
            for entry in coverage_entries:
                reason = str(entry.get("recovery_reason") or "")
                if reason:
                    recovery_reasons[reason] += 1
                tiers = entry.get("source_tiers") or {}
                if int(tiers.get("fallback") or 0):
                    corroboration_candidates += 1
                    corroborated += bool(entry.get("corroborated"))
        elif event_type == "child_query_guard":
            child_queries += 1
            drifted_child_queries += data.get("state") == "rewritten"
        elif event_type == "compression":
            compression_events += 1
            compression_rescues += bool(data.get("rescue_used"))
        elif event_type == "stage_timing":
            stage_timings[str(data.get("stage") or "unknown")] += float(
                data.get("duration_seconds") or 0
            )
        elif event_type == "job_completed":
            final = data

    total_tiered_sources = sum(
        count for tier, count in source_tiers.items() if tier != "reject"
    )
    high_quality_sources = (
        source_tiers.get("primary", 0) + source_tiers.get("reputable", 0)
    )
    requested = float(
        final.get("requested_research_duration_seconds")
        or budget.get("requested_duration_seconds")
        or 0
    )
    estimated = float(
        final.get("estimated_research_duration_seconds")
        or budget.get("estimated_research_seconds")
        or 0
    )
    actual = float(final.get("actual_research_duration_seconds") or 0)
    estimated_stages = budget.get("estimated_stage_seconds") or {}
    ready_aspects = sum(
        entry.get("state") == "evidence_ready" for entry in coverage_entries
    )

    return {
        "trajectory": path.name,
        "job_id": events[0].get("job_id") if events else "",
        "status": final.get("status", "incomplete"),
        "duration_seconds": round(float(final.get("duration_ms") or 0) / 1000, 3),
        "tree_policy": policy.get("tree_policy", ""),
        "branch_mode": policy.get("branch_mode", ""),
        "branch_count": sum(
            event.get("type") == "deep_research_branch"
            and (event.get("data") or {}).get("state") == "completed"
            for event in events
        ),
        "sub_query_count": counts["sub_query"],
        "unique_query_count": len(queries),
        "search_passes": counts["search_results"],
        "selector_calls": sum(
            (event.get("data") or {}).get("selector_mode") == "llm"
            for event in events
            if event.get("type") == "source_selection"
        ),
        "selected_url_count": len(selected_urls),
        "rejected_url_count": len(rejected_urls),
        "source_tiers": dict(source_tiers),
        "accepted_scrapes": accepted_scrapes,
        "rejected_scrapes": rejected_scrapes,
        "max_active_branches": max_active,
        "source_count": int(final.get("source_count") or 0),
        "aspect_count": len(coverage_entries),
        "ready_aspect_count": ready_aspects,
        "aspect_coverage_rate": round(
            ready_aspects / len(coverage_entries), 4
        ) if coverage_entries else 0.0,
        "recovery_reasons": dict(recovery_reasons),
        "primary_reputable_ratio": round(
            high_quality_sources / total_tiered_sources, 4
        ) if total_tiered_sources else 0.0,
        "corroboration_rate": round(
            corroborated / corroboration_candidates, 4
        ) if corroboration_candidates else 0.0,
        "child_query_drift_rate": round(
            drifted_child_queries / child_queries, 4
        ) if child_queries else 0.0,
        "compression_rescue_rate": round(
            compression_rescues / compression_events, 4
        ) if compression_events else 0.0,
        "requested_research_seconds": requested,
        "estimated_research_seconds": estimated,
        "actual_research_seconds": actual,
        "duration_estimation_error_rate": round(
            abs(actual - estimated) / requested, 4
        ) if requested else 0.0,
        "stage_timings_seconds": {
            key: round(value, 3) for key, value in stage_timings.items()
        },
        "stage_estimation_error_seconds": {
            stage: round(
                float(stage_timings.get(stage, 0.0)) - float(estimated), 3
            )
            for stage, estimated in estimated_stages.items()
        },
        "queries": queries,
        "selected_urls": sorted(selected_urls),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "trajectory", "policy", "branches", "queries", "searches", "sources",
        "scrapes", "selector", "coverage", "quality", "duration", "status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            row["trajectory"],
            f"{row['tree_policy']}/{row['branch_mode']}",
            str(row["branch_count"]),
            str(row["unique_query_count"]),
            str(row["search_passes"]),
            str(row["source_count"]),
            f"{row['accepted_scrapes']}/{row['rejected_scrapes']}",
            str(row["selector_calls"]),
            (
                f"{row['ready_aspect_count']}/{row['aspect_count']}"
                if row["aspect_count"]
                else "-"
            ),
            f"{row['primary_reputable_ratio']:.0%}",
            f"{row['duration_seconds']:.1f}s",
            row["status"],
        ]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        else:
            files.append(path)
    rows = [summarize(path) for path in files]
    if args.as_json:
        print(json.dumps(rows, indent=2))
    else:
        print(markdown_table(rows))


if __name__ == "__main__":
    main()
