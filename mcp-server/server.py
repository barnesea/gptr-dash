"""
GPT Researcher MCP Server

This script implements an MCP server for GPT Researcher, allowing AI assistants
to conduct web research and generate reports via the MCP protocol.
"""

import os
import sys
import uuid
import logging
import asyncio
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import quote
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastmcp import Context, FastMCP
from fastmcp.server.dependencies import get_http_request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_researcher import GPTResearcher
from gpt_researcher.config import Config
from gpt_researcher.utils.enum import ReportType
from gpt_researcher.utils.research_budget import (
    build_research_policy,
    execution_policy,
    report_word_target,
    validate_research_duration,
)

# Load environment variables
load_dotenv()

from utils import (
    research_store,
    create_success_response, 
    handle_exception,
    get_researcher_by_id, 
    format_sources_for_response,
    source_urls_from_sources,
    format_context_with_sources, 
    store_research_results,
    create_research_prompt
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
)

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="GPT Researcher"
)

# Initialize researchers dictionary
if not hasattr(mcp, "researchers"):
    mcp.researchers = {}

_deep_research_admission_lock = asyncio.Lock()
_deep_research_active_jobs = 0
_deep_research_last_started_at: float | None = None


async def _mcp_info(ctx: Context | None, message: str) -> None:
    if ctx is None:
        return
    try:
        await ctx.info(message)
    except Exception as exc:
        logger.debug("Failed to send MCP info notification: %s", exc)


async def _mcp_progress(
    ctx: Context | None,
    progress: int,
    total: int,
    message: str | None = None,
    done: bool = False,
) -> None:
    await _openwebui_status(ctx, message or "Working", done, progress, total)

    if ctx is None:
        return
    try:
        await ctx.report_progress(
            progress=max(0, progress),
            total=max(1, total),
            message=message,
        )
    except Exception as exc:
        logger.debug("Failed to send MCP progress notification: %s", exc)


def _openwebui_events_enabled() -> bool:
    return os.getenv("OPENWEBUI_EVENTS_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_flag(name: str, default: bool) -> bool:
    """Read a conventional boolean environment switch without raising."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _get_openwebui_request_headers() -> dict[str, str]:
    try:
        request = get_http_request()
    except Exception:
        return {}

    return dict(request.headers) if getattr(request, "headers", None) else {}


def _get_openwebui_event_context() -> dict[str, str]:
    headers = _get_openwebui_request_headers()
    chat_header = os.getenv("OPENWEBUI_CHAT_ID_HEADER", "X-OpenWebUI-Chat-Id").lower()
    message_header = os.getenv("OPENWEBUI_MESSAGE_ID_HEADER", "X-OpenWebUI-Message-Id").lower()

    chat_id = headers.get(chat_header, "").strip()
    message_id = headers.get(message_header, "").strip()
    if not chat_id or not message_id:
        return {}

    token = (
        os.getenv("OPENWEBUI_EVENT_API_KEY", "").strip()
        or os.getenv("OPENWEBUI_API_KEY", "").strip()
    )
    authorization = f"Bearer {token}" if token else headers.get("authorization", "").strip()
    if not authorization:
        return {}

    base_url = (
        os.getenv("OPENWEBUI_BASE_URL", "").strip()
        or os.getenv("OPENWEBUI_URL", "").strip()
        or "http://open-webui:9090"
    )

    return {
        "authorization": authorization,
        "base_url": base_url.rstrip("/"),
        "chat_id": chat_id,
        "message_id": message_id,
    }


async def _openwebui_event(ctx: Context | None, event_type: str, data: dict[str, Any]) -> None:
    if not _openwebui_events_enabled():
        return

    event_context = _get_openwebui_event_context()
    if not event_context:
        return

    chat_id = quote(event_context["chat_id"], safe="")
    message_id = quote(event_context["message_id"], safe="")
    url = (
        f"{event_context['base_url']}/api/v1/chats/"
        f"{chat_id}/messages/{message_id}/event"
    )
    timeout = float(os.getenv("OPENWEBUI_EVENT_TIMEOUT_S", "5"))

    def post_event() -> None:
        response = requests.post(
            url,
            headers={"Authorization": event_context["authorization"]},
            json={"type": event_type, "data": data},
            timeout=timeout,
        )
        response.raise_for_status()

    try:
        await asyncio.to_thread(post_event)
    except Exception as exc:
        logger.debug("Failed to emit OpenWebUI %s event: %s", event_type, exc)


async def _openwebui_status(
    ctx: Context | None,
    description: str,
    done: bool = False,
    progress: int | None = None,
    total: int | None = None,
) -> None:
    data: dict[str, Any] = {
        "action": "gptr_deep_research",
        "description": description,
        "done": done,
    }
    if progress is not None:
        data["progress"] = max(0, progress)
    if total is not None:
        data["total"] = max(1, total)

    await _openwebui_event(ctx, "status", data)


def _format_deep_research_progress(progress: Any) -> tuple[int, int, str]:
    total_breadth = max(
        1,
        int(getattr(progress, "total_queries", 0) or 0),
        int(getattr(progress, "total_breadth", 0) or 0),
    )
    total_depth = max(1, int(getattr(progress, "total_depth", 1) or 1))
    current_depth = min(
        total_depth,
        max(1, int(getattr(progress, "current_depth", 1) or 1)),
    )
    completed_queries = max(0, int(getattr(progress, "completed_queries", 0) or 0))

    total = total_depth * total_breadth
    current = min(total, ((current_depth - 1) * total_breadth) + completed_queries)

    current_query = getattr(progress, "current_query", None)
    message = (
        f"Deep research depth {current_depth}/{total_depth}, "
        f"queries {min(completed_queries, total_breadth)}/{total_breadth}"
    )
    if current_query:
        message = f"{message}: {str(current_query)[:180]}"

    return current, total, message


def _build_deep_research_progress_callback(ctx: Context | None):
    if ctx is None:
        return None

    seen: set[tuple[int, int, str]] = set()
    tasks: list[asyncio.Task] = []

    def on_progress(progress: Any) -> None:
        current, total, message = _format_deep_research_progress(progress)
        signature = (current, total, message)
        if signature in seen:
            return
        seen.add(signature)
        tasks.append(asyncio.create_task(_mcp_progress(ctx, current, total, message)))

    on_progress.tasks = tasks
    return on_progress


async def _drain_progress_tasks(on_progress) -> None:
    tasks = getattr(on_progress, "tasks", None)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def get_deep_research_limit() -> int:
    """Return max concurrent deep_research jobs. Zero disables the active limit."""
    raw_limit = os.getenv("MCP_MAX_CONCURRENT_DEEP_RESEARCH", "1").strip()
    try:
        limit = int(raw_limit)
    except ValueError:
        logger.warning(
            "Invalid MCP_MAX_CONCURRENT_DEEP_RESEARCH=%r, defaulting to 1",
            raw_limit,
        )
        return 1

    return max(limit, 0)


def get_deep_research_cooldown_seconds() -> int:
    """Return the global cooldown between accepted deep_research calls."""
    raw_cooldown = os.getenv("MCP_DEEP_RESEARCH_COOLDOWN_SECONDS", "300").strip()
    try:
        cooldown = int(raw_cooldown)
    except ValueError:
        logger.warning(
            "Invalid MCP_DEEP_RESEARCH_COOLDOWN_SECONDS=%r, defaulting to 300",
            raw_cooldown,
        )
        return 300

    return max(cooldown, 0)


def _get_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r, defaulting to %s", name, raw_value, default)
        return default
    return max(1, value)


def get_deep_research_max_depth() -> int:
    """Return the maximum recursive depth exposed through the MCP tool."""
    return _get_positive_int_env("MCP_DEEP_RESEARCH_MAX_DEPTH", 2)


def get_deep_research_max_breadth() -> int:
    """Return the maximum branch count exposed through the MCP tool."""
    return _get_positive_int_env("MCP_DEEP_RESEARCH_MAX_BREADTH", 3)


def validate_deep_research_budget(depth: int, breadth: int) -> Dict[str, Any] | None:
    """Return a structured error when a requested research budget is unsafe."""
    max_depth = get_deep_research_max_depth()
    max_breadth = get_deep_research_max_breadth()
    if 1 <= depth <= max_depth and 1 <= breadth <= max_breadth:
        return None
    return {
        "status": "invalid_research_budget",
        "message": (
            f"depth must be between 1 and {max_depth}, and breadth must be between "
            f"1 and {max_breadth}."
        ),
        "requested_depth": depth,
        "requested_breadth": breadth,
        "max_depth": max_depth,
        "max_breadth": max_breadth,
    }


def validate_research_duration_input(
    research_duration_seconds: Any,
) -> tuple[int | None, Dict[str, Any] | None]:
    """Validate the only public deep-research budget input."""
    try:
        return validate_research_duration(research_duration_seconds), None
    except ValueError as exc:
        return None, {
            "status": "invalid_research_duration",
            "message": str(exc),
            "requested_research_duration_seconds": research_duration_seconds,
            "min_research_duration_seconds": 15,
            "max_research_duration_seconds": 600,
        }


def _deep_research_rate_limited_response(wait_seconds: int) -> Dict[str, Any]:
    return {
        "status": "rate_limited",
        "message": (
            "The deep_research tool has been called too frequently. "
            f"Please wait {wait_seconds} seconds before calling it again."
        ),
        "retry_after_seconds": wait_seconds,
    }


def _deep_research_busy_response(active_jobs: int, limit: int) -> Dict[str, Any]:
    return {
        "status": "busy",
        "message": (
            "A deep_research job is already running. Do not call this tool again "
            "until the current job finishes or the cooldown has expired."
        ),
        "active_jobs": active_jobs,
        "max_concurrent_jobs": limit,
    }


async def admit_deep_research_call() -> Dict[str, Any] | None:
    """Admit a deep_research call or return a refusal response without queueing."""
    global _deep_research_active_jobs, _deep_research_last_started_at

    async with _deep_research_admission_lock:
        now = time.monotonic()
        cooldown = get_deep_research_cooldown_seconds()
        if cooldown and _deep_research_last_started_at is not None:
            elapsed = now - _deep_research_last_started_at
            if elapsed < cooldown:
                return _deep_research_rate_limited_response(
                    max(1, int(cooldown - elapsed))
                )

        limit = get_deep_research_limit()
        if limit and _deep_research_active_jobs >= limit:
            return _deep_research_busy_response(_deep_research_active_jobs, limit)

        _deep_research_active_jobs += 1
        _deep_research_last_started_at = now
        return None


async def release_deep_research_call() -> None:
    """Release an admitted deep_research call."""
    global _deep_research_active_jobs

    async with _deep_research_admission_lock:
        _deep_research_active_jobs = max(0, _deep_research_active_jobs - 1)


def recursive_deep_research_enabled() -> bool:
    """Return whether MCP deep_research should use GPT Researcher's recursive mode."""
    return os.getenv("MCP_ENABLE_RECURSIVE_DEEP_RESEARCH", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def check_research_dependency_status() -> Dict[str, Any]:
    """Check the external services used by MCP web research."""
    retriever = os.getenv("RETRIEVER", "tavily")
    scraper = os.getenv("SCRAPER", "bs")
    searx_url = os.getenv("SEARX_URL") or os.getenv("SEARXNG_URL", "")
    crawl4ai_url = os.getenv("CRAWL4AI_BASE_URL", "")

    checks: Dict[str, Any] = {
        "config": {
            "retriever": retriever,
            "scraper": scraper,
            "searx_url": searx_url,
            "crawl4ai_url": crawl4ai_url,
            "crawl4ai_token_configured": bool(os.getenv("CRAWL4AI_API_TOKEN")),
            "mcp_transport": os.getenv("MCP_TRANSPORT", "stdio"),
            "mcp_path": os.getenv("MCP_PATH", "/mcp"),
            "recursive_deep_research_enabled": recursive_deep_research_enabled(),
            "mcp_max_concurrent_deep_research": get_deep_research_limit(),
            "mcp_deep_research_active_jobs": _deep_research_active_jobs,
            "mcp_deep_research_cooldown_seconds": get_deep_research_cooldown_seconds(),
            "mcp_deep_research_max_depth": get_deep_research_max_depth(),
            "mcp_deep_research_max_breadth": get_deep_research_max_breadth(),
            "research_duration_controller_mode": os.getenv(
                "RESEARCH_DURATION_CONTROLLER_MODE", "off"
            ),
            "research_duration_default_seconds": os.getenv(
                "RESEARCH_DURATION_DEFAULT_SECONDS", "60"
            ),
            "research_budget_calibration_enabled": os.getenv(
                "RESEARCH_BUDGET_CALIBRATION_ENABLED", "true"
            ),
            "deep_research_adaptive_compression": os.getenv(
                "DEEP_RESEARCH_ADAPTIVE_COMPRESSION", "false"
            ),
            "deep_research_fallback_corroboration": os.getenv(
                "DEEP_RESEARCH_FALLBACK_CORROBORATION", "2"
            ),
            "deep_research_fallback_corroboration_enabled": os.getenv(
                "DEEP_RESEARCH_FALLBACK_CORROBORATION_ENABLED", "false"
            ),
            "deep_research_pdf_connect_timeout_seconds": os.getenv(
                "DEEP_RESEARCH_PDF_CONNECT_TIMEOUT_SECONDS", "3"
            ),
            "deep_research_pdf_total_timeout_seconds": os.getenv(
                "DEEP_RESEARCH_PDF_TOTAL_TIMEOUT_SECONDS", "8"
            ),
            "deep_research_pdf_max_bytes": os.getenv(
                "DEEP_RESEARCH_PDF_MAX_BYTES", str(32 * 1024 * 1024)
            ),
            "recursive_deep_research_breadth": os.getenv("DEEP_RESEARCH_BREADTH", ""),
            "recursive_deep_research_depth": os.getenv("DEEP_RESEARCH_DEPTH", ""),
            "recursive_deep_research_concurrency": os.getenv("DEEP_RESEARCH_CONCURRENCY", ""),
            "deep_research_focused_retrieval": os.getenv("DEEP_RESEARCH_FOCUSED_RETRIEVAL", ""),
            "deep_research_tree_policy": os.getenv("DEEP_RESEARCH_TREE_POLICY", ""),
            "deep_research_branch_mode": os.getenv("DEEP_RESEARCH_BRANCH_MODE", ""),
            "deep_research_max_deepened_branches": os.getenv("DEEP_RESEARCH_MAX_DEEPENED_BRANCHES", ""),
            "deep_research_min_deepening_score": os.getenv("DEEP_RESEARCH_MIN_DEEPENING_SCORE", ""),
            "deep_research_source_standards": os.getenv("DEEP_RESEARCH_SOURCE_STANDARDS", ""),
            "deep_research_direct_url_seed": os.getenv("DEEP_RESEARCH_DIRECT_URL_SEED", ""),
            "retrieval_pipeline_mode": os.getenv(
                "RETRIEVAL_PIPELINE_MODE", "legacy"
            ),
            "source_evidence_judge_mode": os.getenv(
                "SOURCE_EVIDENCE_JUDGE_MODE", "all"
            ),
            "source_evidence_judge_fallback": os.getenv(
                "SOURCE_EVIDENCE_JUDGE_FALLBACK", "hybrid"
            ),
            "canonical_content_resolution": os.getenv(
                "CANONICAL_CONTENT_RESOLUTION", "true"
            ),
            "source_selector_mode": os.getenv("SOURCE_SELECTOR_MODE", ""),
            "research_trajectory_enabled": os.getenv("RESEARCH_TRAJECTORY_ENABLED", ""),
            "research_trajectory_dir": os.getenv("RESEARCH_TRAJECTORY_DIR", ""),
        },
        "searx": {"enabled": retriever in {"searx", "searxng"}},
        "crawl4ai": {"enabled": scraper == "crawl4ai"},
    }

    if checks["searx"]["enabled"]:
        try:
            response = requests.get(
                f"{searx_url.rstrip('/')}/search",
                params={"q": "example domain", "format": "json"},
                timeout=10,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            checks["searx"].update(
                {"ok": bool(results), "status_code": response.status_code, "result_count": len(results)}
            )
        except Exception as exc:
            checks["searx"].update({"ok": False, "error": str(exc)})

    if checks["crawl4ai"]["enabled"]:
        try:
            from gpt_researcher.scraper.crawl4ai.crawl4ai import Crawl4AIScraper

            content, _images, title = Crawl4AIScraper("https://example.com").scrape()
            checks["crawl4ai"].update(
                {
                    "ok": bool(content),
                    "title": title,
                    "content_length": len(content),
                }
            )
        except Exception as exc:
            checks["crawl4ai"].update({"ok": False, "error": str(exc)})

    dependency_checks = [
        check for name, check in checks.items()
        if name != "config" and check.get("enabled")
    ]
    checks["ok"] = all(check.get("ok") for check in dependency_checks)
    return checks


@mcp.tool()
async def check_research_dependencies() -> Dict[str, Any]:
    """
    Check the configured research dependency path used by deep_research.

    This verifies SearXNG search and Crawl4AI scraping from inside the MCP server
    process, which catches Docker DNS, auth, and service health issues before a
    full research job returns empty context.
    """
    return create_success_response(
        await asyncio.to_thread(check_research_dependency_status)
    )


@mcp.resource("research://{topic}")
async def research_resource(topic: str) -> str:
    """
    Provide research context for a given topic directly as a resource.
    
    This allows LLMs to access web-sourced information without explicit function calls.
    
    Args:
        topic: The research topic or query
        
    Returns:
        String containing the research context with source information
    """
    # Check if we've already researched this topic
    if topic in research_store:
        logger.info(f"Returning cached research for topic: {topic}")
        return research_store[topic]["context"]
    
    # If not, conduct the research
    logger.info(f"Conducting new research for resource on topic: {topic}")
    
    # Initialize GPT Researcher
    researcher = GPTResearcher(topic)
    
    try:
        # Conduct the research
        await researcher.conduct_research()
        
        # Get the context and sources
        context = researcher.get_research_context()
        sources = researcher.get_research_sources()
        source_urls = source_urls_from_sources(sources)
        
        # Format with sources included
        formatted_context = format_context_with_sources(topic, context, sources)
        
        # Store for future use
        store_research_results(topic, context, sources, source_urls, formatted_context)
        
        return formatted_context
    except Exception as e:
        return f"Error conducting research on '{topic}': {str(e)}"


@mcp.tool()
async def deep_research(
    query: str,
    ctx: Context,
    research_duration_seconds: int = 60,
) -> Dict[str, Any]:
    """
    Conduct bounded, cited web research using GPT Researcher.
    Use this tool for explicit deep-research requests, any request that supplies
    a duration, current or niche subjects, comparisons, and source-heavy
    synthesis. Always honor an explicit deep-research request. For stable,
    simple factual questions, prefer quick_search or normal model knowledge.
    
    Args:
        query: The research query or topic
        research_duration_seconds: Best-effort research time target, excluding
            final report generation. Choose 15-29 seconds for a focused answer,
            30-59 for a small multi-part question, 60-119 for normal deep
            research, 120-239 for broad or versioned topics, and 240-600 for
            extensive synthesis. The backend converts the duration into bounded
            coverage, recovery, corroboration, and deepening limits.
        
    Returns:
        Dict containing a cited, final GPT Researcher report and its sources.  On
        success, the caller MUST output the ``report`` field verbatim. Do not
        summarize, reinterpret, expand, or supplement it with model knowledge.
        Preserve every evidence limitation. Do not run extra quick searches or
        call write_report afterwards: this tool already completes both research
        and report writing in one bounded operation.
    """
    duration, budget_error = validate_research_duration_input(
        research_duration_seconds
    )
    if budget_error is not None:
        await _mcp_info(ctx, budget_error["message"])
        await _openwebui_status(ctx, budget_error["message"], done=True)
        return budget_error

    await _mcp_info(
        ctx,
        f"Starting deep research (target={duration}s, report time excluded): {query}",
    )
    await _mcp_progress(ctx, 0, 1, "Deep research queued")

    refusal = await admit_deep_research_call()
    if refusal is not None:
        logger.info("Rejecting deep_research call for query %r: %s", query, refusal["message"])
        await _mcp_info(ctx, refusal["message"])
        await _openwebui_status(ctx, refusal["message"], done=True)
        return refusal

    try:
        return await _run_deep_research(
            query, ctx, research_duration_seconds=duration or 60
        )
    finally:
        await release_deep_research_call()


async def _run_deep_research(
    query: str,
    ctx: Context | None = None,
    *,
    research_duration_seconds: int = 60,
) -> Dict[str, Any]:
    """Run a deep research job after concurrency admission."""
    cfg = Config()
    calculated_policy = build_research_policy(research_duration_seconds, cfg)
    active_policy = execution_policy(calculated_policy, cfg)
    logger.info(
        "Conducting research with target=%ss, mode=%s, aspects=%s: %s...",
        research_duration_seconds,
        calculated_policy.controller_mode,
        active_policy.aspect_count,
        query,
    )
    await _mcp_progress(
        ctx,
        0,
        1,
        f"Deep research started (target={research_duration_seconds}s)",
    )
    
    # Generate a unique ID for this research session
    research_id = str(uuid.uuid4())
    
    report_type = (
        ReportType.DeepResearch.value
        if recursive_deep_research_enabled()
        else ReportType.ResearchReport.value
    )

    # Initialize GPT Researcher
    researcher = GPTResearcher(
        query,
        report_type=report_type,
        trajectory_id=research_id,
        research_policy=active_policy,
    )
    if report_type == ReportType.DeepResearch.value and researcher.deep_researcher:
        researcher.deep_researcher.depth = active_policy.max_depth
        researcher.deep_researcher.breadth = active_policy.aspect_count
        researcher.deep_researcher.max_deepened_branches = (
            active_policy.max_deepened_branches
        )
    calculated_policy_payload = calculated_policy.to_dict()
    execution_policy_payload = active_policy.to_dict()
    researcher.trace_event(
        "research_budget",
        {
            "controller_mode": calculated_policy.controller_mode,
            "requested_duration_seconds": research_duration_seconds,
            "calculated_policy": calculated_policy_payload,
            "execution_policy": execution_policy_payload,
            "policy_match": (
                calculated_policy.policy_signature
                == active_policy.policy_signature
            ),
        },
    )
    on_progress = _build_deep_research_progress_callback(ctx)
    
    # Start research
    try:
        research_started_at = time.perf_counter()
        await researcher.conduct_research(on_progress=on_progress)
        actual_research_duration = time.perf_counter() - research_started_at
        await _drain_progress_tasks(on_progress)
        mcp.researchers[research_id] = researcher
        logger.info(f"Research completed for ID: {research_id}")
        await _mcp_info(ctx, "Deep research completed; preparing the cited report.")
        
        # Get the research context and sources
        context = researcher.get_research_context()
        sources = researcher.get_research_sources()
        source_urls = source_urls_from_sources(sources)
        
        # Store in the research store for the resource API
        store_research_results(query, context, sources, source_urls)

        retrieved_source_urls = sorted(
            {
                str(url)
                for item in researcher.coverage_ledger
                for url in item.get("retrieved_urls", [])
                if str(url).strip()
            }
        )
        excluded_source_urls = [
            url for url in retrieved_source_urls if url not in source_urls
        ]
        response: Dict[str, Any] = {
            "answer_ready": False,
            "presentation_instruction": (
                "Wait for answer_ready=true, then output only the report field "
                "verbatim. Never add facts from model knowledge."
            ),
            "query": query,
            "report_type": report_type,
            "recursive_deep_research_enabled": recursive_deep_research_enabled(),
            "requested_research_duration_seconds": research_duration_seconds,
            "estimated_research_duration_seconds": (
                calculated_policy.estimated_research_seconds
            ),
            "actual_research_duration_seconds": round(
                actual_research_duration, 3
            ),
            "report_generation_duration_seconds": None,
            "calculated_policy": calculated_policy.to_dict(),
            "execution_policy": active_policy.to_dict(),
            "calibration_source": calculated_policy.calibration_source,
            "calibration_sample_count": calculated_policy.calibration_sample_count,
            "source_count": len(sources),
            "sources": format_sources_for_response(sources),
            "source_urls": source_urls,
            "retrieved_source_count": len(retrieved_source_urls),
            "retrieved_source_urls": retrieved_source_urls,
            "excluded_source_count": len(excluded_source_urls),
            "excluded_source_urls": excluded_source_urls,
            "coverage_ledger": researcher.coverage_ledger,
            "executed_work_units": (
                researcher.deep_researcher._executed_work_units
                if researcher.deep_researcher
                else active_policy.work_units
            ),
        }
        if researcher.deep_researcher:
            response["tree_policy"] = researcher.deep_researcher.tree_policy
            response["branch_mode"] = researcher.deep_researcher.branch_mode
        candidate_snapshot = getattr(
            researcher, "candidate_ledger_snapshot", None
        )
        if isinstance(candidate_snapshot, dict):
            candidate_entries = candidate_snapshot.get("candidates") or []
            response["retrieval_pipeline"] = {
                "version": candidate_snapshot.get(
                    "pipeline_version", "v2"
                ),
                "candidate_count": candidate_snapshot.get(
                    "candidate_count", len(candidate_entries)
                ),
                "attempted_queries": candidate_snapshot.get(
                    "attempted_queries", []
                ),
                "preliminary_reuse_count": sum(
                    any(
                        discovery.get("stage") == "preliminary"
                        for discovery in entry.get("discoveries", [])
                    )
                    and bool(entry.get("assigned_aspects"))
                    for entry in candidate_entries
                ),
                "resolver_attempt_count": sum(
                    len(entry.get("fetch_attempts", []))
                    for entry in candidate_entries
                ),
                "evidence_judgment_count": sum(
                    len(entry.get("judgments", []))
                    for entry in candidate_entries
                ),
                "judge_fallback_count": sum(
                    bool(judgment.get("fallback"))
                    for entry in candidate_entries
                    for judgment in entry.get("judgments", [])
                ),
            }
        if researcher.trajectory:
            response["trajectory_id"] = researcher.trajectory.job_id
        # A UUID is too easy for a chat model to transcribe incorrectly.  The
        # default response is self-contained, while an explicit diagnostics
        # switch retains the old multi-tool workflow for debugging clients.
        if _env_flag("MCP_DEEP_RESEARCH_RETURN_REPORT", True):
            await _mcp_progress(ctx, 1, 1, "Writing cited research report")
            target_words = report_word_target(research_duration_seconds)
            researcher.cfg.total_words = target_words
            researcher.trace_event("report_generation", {"state": "started"})
            report_started_at = time.perf_counter()
            report = await researcher.write_report()
            report_duration = time.perf_counter() - report_started_at
            researcher.trace_event("report_generation", {
                "state": "completed",
                "duration_seconds": round(report_duration, 3),
                "report_chars": len(report or ""),
            })
            researcher.trace_event(
                "stage_timing",
                {
                    "stage": "reporting",
                    "duration_seconds": round(report_duration, 3),
                    "estimated_seconds": calculated_policy.estimated_stage_seconds.get(
                        "reporting", 0.0
                    ),
                },
            )
            response["report"] = report
            response["report_generated"] = True
            response["report_word_target"] = target_words
            response["report_generation_duration_seconds"] = round(
                report_duration, 3
            )
            response["answer_ready"] = True
            response["presentation_instruction"] = (
                "MANDATORY: output only the report field verbatim, with no "
                "preface, summary, reinterpretation, expansion, or additional "
                "facts. Preserve all evidence limitations. Do not use model "
                "knowledge to fill gaps."
            )
        else:
            response["context"] = context
            response["report_generated"] = False
            response["answer_ready"] = False
        if _env_flag("MCP_EXPOSE_RESEARCH_ID", False):
            response["research_id"] = research_id
        evidence_status = (
            "success"
            if sources and str(context or "").strip()
            else "insufficient_evidence"
        )
        if researcher.trajectory:
            researcher.trajectory.finalize(evidence_status, {
                "source_count": len(sources),
                "context_chars": len(str(context or "")),
                "report_generated": response["report_generated"],
                "actual_research_duration_seconds": round(
                    actual_research_duration, 3
                ),
                "report_generation_duration_seconds": response[
                    "report_generation_duration_seconds"
                ],
                "requested_research_duration_seconds": research_duration_seconds,
                "estimated_research_duration_seconds": (
                    calculated_policy.estimated_research_seconds
                ),
                "calculated_policy_signature": (
                    calculated_policy.policy_signature
                ),
                "execution_policy_signature": active_policy.policy_signature,
                "executed_work_units": response["executed_work_units"],
                "controller_mode": calculated_policy.controller_mode,
            })
        await _mcp_progress(ctx, 1, 1, "Deep research report completed", done=True)
        return {"status": evidence_status, **response}
    except Exception as e:
        await _drain_progress_tasks(on_progress)
        if researcher.trajectory:
            researcher.trajectory.finalize("error", {"error": str(e)})
        await _mcp_info(ctx, f"Deep research failed: {e}")
        await _openwebui_status(ctx, f"Deep research failed: {e}", done=True)
        return handle_exception(e, "Research")


@mcp.tool()
async def quick_search(query: str) -> Dict[str, Any]:
    """
    Perform a quick web search on a given query and return search results with snippets.
    This optimizes for speed over quality. Use it for stable, simple factual
    questions or a quick freshness check. Use deep_research when the user
    explicitly requests research, supplies a duration, needs comparison or
    synthesis, or asks about a current/niche topic.
    
    Args:
        query: The search query
        
    Returns:
        Dict containing search results and snippets
    """
    logger.info(f"Performing quick search on query: {query}...")
    
    # Generate a unique ID for this search session
    search_id = str(uuid.uuid4())
    
    # Initialize GPT Researcher
    researcher = GPTResearcher(query)
    
    try:
        # Perform quick search
        search_results = await researcher.quick_search(query=query)
        mcp.researchers[search_id] = researcher
        logger.info(f"Quick search completed for ID: {search_id}")
        
        return create_success_response({
            "search_id": search_id,
            "query": query,
            "result_count": len(search_results) if search_results else 0,
            "search_results": search_results
        })
    except Exception as e:
        return handle_exception(e, "Quick search")


@mcp.tool()
async def write_report(research_id: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Legacy diagnostic tool for generating a report from an explicitly exposed
    research ID. Normal deep_research calls already return the cited report;
    do not call this after deep_research.
    
    Args:
        research_id: The ID of the research session from deep_research
        custom_prompt: Optional custom prompt for report generation
        
    Returns:
        Dict containing the report content and metadata
    """
    success, researcher, error = get_researcher_by_id(mcp.researchers, research_id)
    if not success:
        return error
    
    logger.info(f"Generating report for research ID: {research_id}")
    
    try:
        # Generate report
        report = await researcher.write_report(custom_prompt=custom_prompt)
        
        # Get additional information
        sources = researcher.get_research_sources()
        costs = researcher.get_costs()
        
        return create_success_response({
            "report": report,
            "source_count": len(sources),
            "costs": costs
        })
    except Exception as e:
        return handle_exception(e, "Report generation")


@mcp.tool()
async def get_research_sources(research_id: str) -> Dict[str, Any]:
    """
    Get the sources used in the research.
    
    Args:
        research_id: The ID of the research session
        
    Returns:
        Dict containing the research sources
    """
    success, researcher, error = get_researcher_by_id(mcp.researchers, research_id)
    if not success:
        return error
    
    sources = researcher.get_research_sources()
    source_urls = source_urls_from_sources(sources)
    
    return create_success_response({
        "sources": format_sources_for_response(sources),
        "source_urls": source_urls
    })


@mcp.tool()
async def get_research_context(research_id: str) -> Dict[str, Any]:
    """
    Get the full context of the research.
    
    Args:
        research_id: The ID of the research session
        
    Returns:
        Dict containing the research context
    """
    success, researcher, error = get_researcher_by_id(mcp.researchers, research_id)
    if not success:
        return error
    
    context = researcher.get_research_context()
    
    return create_success_response({
        "context": context
    })


@mcp.prompt()
def research_query(topic: str, goal: str, report_format: str = "research_report") -> str:
    """
    Create a research query prompt for GPT Researcher.
    
    Args:
        topic: The topic to research
        goal: The goal or specific question to answer
        report_format: The format of the report to generate
        
    Returns:
        A formatted prompt for research
    """
    return create_research_prompt(topic, goal, report_format)

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "healthy", "service": "mcp-server"})


@mcp.custom_route("/health/dependencies", methods=["GET"])
async def dependency_health_check(request):
    status = await asyncio.to_thread(check_research_dependency_status)
    return JSONResponse(status, status_code=200 if status.get("ok") else 503)

def run_server():
    """Run the MCP server using FastMCP's built-in event loop handling."""
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        return

    # Determine transport based on environment.
    transport = os.getenv("MCP_TRANSPORT")
    port = int(os.getenv("MCP_PORT", "8001"))
    is_docker = bool(os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER"))

    if not transport:
        transport = "sse" if is_docker else "stdio"
        if is_docker:
            logger.info("Docker environment detected, defaulting to SSE transport")
    
    transport = transport.lower().replace("_", "-")
    if transport == "http":
        transport = "streamable-http"
    
    # Add startup message
    logger.info(f"Starting GPT Researcher MCP Server with {transport} transport...")
    print(f"🚀 GPT Researcher MCP Server starting with {transport} transport...")
    print("   Check researcher_mcp_server.log for details")

    # Let FastMCP handle the event loop
    try:
        if transport == "stdio":
            logger.info("Using STDIO transport (Claude Desktop compatible)")
            mcp.run(transport="stdio")
        elif transport == "sse":
            mcp.run(transport="sse", host="0.0.0.0", port=port)
        elif transport == "streamable-http":
            mcp_path = os.getenv("MCP_PATH", "/mcp")
            logger.info(f"Using streamable HTTP endpoint at {mcp_path}")
            mcp.run(
                transport="streamable-http",
                host="0.0.0.0",
                port=port,
                path=mcp_path,
            )
        else:
            raise ValueError(f"Unsupported transport: {transport}")
            
        # Note: If we reach here, the server has stopped
        logger.info("MCP Server is running...")
        while True:
            pass  # Keep the process alive
    except Exception as e:
        logger.error(f"Error running MCP server: {str(e)}")
        print(f"❌ MCP Server error: {str(e)}")
        return
        
    print("✅ MCP Server stopped")


if __name__ == "__main__":
    # Use the non-async approach to avoid asyncio nesting issues
    run_server()
