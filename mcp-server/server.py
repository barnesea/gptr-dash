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
from gpt_researcher.utils.enum import ReportType

# Load environment variables
load_dotenv()

from utils import (
    research_store,
    create_success_response, 
    handle_exception,
    get_researcher_by_id, 
    format_sources_for_response,
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
            "recursive_deep_research_breadth": os.getenv("DEEP_RESEARCH_BREADTH", ""),
            "recursive_deep_research_depth": os.getenv("DEEP_RESEARCH_DEPTH", ""),
            "recursive_deep_research_concurrency": os.getenv("DEEP_RESEARCH_CONCURRENCY", ""),
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
        source_urls = researcher.get_source_urls()
        
        # Format with sources included
        formatted_context = format_context_with_sources(topic, context, sources)
        
        # Store for future use
        store_research_results(topic, context, sources, source_urls, formatted_context)
        
        return formatted_context
    except Exception as e:
        return f"Error conducting research on '{topic}': {str(e)}"


@mcp.tool()
async def deep_research(query: str, ctx: Context) -> Dict[str, Any]:
    """
    Conduct a web deep research on a given query using GPT Researcher. 
    Use this tool when you need time-sensitive, real-time information like stock prices, news, people, specific knowledge, etc.
    
    Args:
        query: The research query or topic
        
    Returns:
        Dict containing research status, ID, and the actual research context and sources
        that can be used directly by LLMs for context enrichment
    """
    await _mcp_info(ctx, f"Starting deep research: {query}")
    await _mcp_progress(ctx, 0, 1, "Deep research queued")

    refusal = await admit_deep_research_call()
    if refusal is not None:
        logger.info("Rejecting deep_research call for query %r: %s", query, refusal["message"])
        await _mcp_info(ctx, refusal["message"])
        await _openwebui_status(ctx, refusal["message"], done=True)
        return refusal

    try:
        return await _run_deep_research(query, ctx)
    finally:
        await release_deep_research_call()


async def _run_deep_research(query: str, ctx: Context | None = None) -> Dict[str, Any]:
    """Run a deep research job after concurrency admission."""
    logger.info(f"Conducting research on query: {query}...")
    await _mcp_progress(ctx, 0, 1, "Deep research started")
    
    # Generate a unique ID for this research session
    research_id = str(uuid.uuid4())
    
    report_type = (
        ReportType.DeepResearch.value
        if recursive_deep_research_enabled()
        else ReportType.ResearchReport.value
    )

    # Initialize GPT Researcher
    researcher = GPTResearcher(query, report_type=report_type)
    on_progress = _build_deep_research_progress_callback(ctx)
    
    # Start research
    try:
        await researcher.conduct_research(on_progress=on_progress)
        await _drain_progress_tasks(on_progress)
        mcp.researchers[research_id] = researcher
        logger.info(f"Research completed for ID: {research_id}")
        await _mcp_progress(ctx, 1, 1, "Deep research completed", done=True)
        await _mcp_info(ctx, f"Deep research completed: {research_id}")
        
        # Get the research context and sources
        context = researcher.get_research_context()
        sources = researcher.get_research_sources()
        source_urls = researcher.get_source_urls()
        
        # Store in the research store for the resource API
        store_research_results(query, context, sources, source_urls)
        
        return create_success_response({
            "research_id": research_id,
            "query": query,
            "report_type": report_type,
            "recursive_deep_research_enabled": recursive_deep_research_enabled(),
            "source_count": len(sources),
            "context": context,
            "sources": format_sources_for_response(sources),
            "source_urls": source_urls
        })
    except Exception as e:
        await _drain_progress_tasks(on_progress)
        await _mcp_info(ctx, f"Deep research failed: {e}")
        await _openwebui_status(ctx, f"Deep research failed: {e}", done=True)
        return handle_exception(e, "Research")


@mcp.tool()
async def quick_search(query: str) -> Dict[str, Any]:
    """
    Perform a quick web search on a given query and return search results with snippets.
    This optimizes for speed over quality and is useful when an LLM doesn't need in-depth
    information on a topic.
    
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
    Generate a report based on previously conducted research.
    
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
    source_urls = researcher.get_source_urls()
    
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
