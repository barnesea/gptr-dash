"""Research conductor skill for GPT Researcher.

This module provides the ResearchConductor class that manages and
coordinates the research process including query planning, web searching,
and context gathering.
"""

import asyncio
import logging
import os
import time

import json_repair

from ..actions.agent_creator import choose_agent
from ..actions.query_processing import get_search_results, plan_research_outline
from ..actions.source_selection import (
    canonical_source_alternatives,
    deterministic_select_sources,
    has_meaningful_query_anchor,
    parse_model_selection,
    post_scrape_integrity_reason,
    should_use_model_selector,
    source_card,
    source_quality_tier,
    source_url,
)
from ..actions.utils import stream_output
from ..document import DocumentLoader, LangChainDocumentLoader, OnlineDocumentLoader
from ..utils.enum import ReportSource, ReportType
from ..utils.logging_config import get_json_handler
from ..utils.llm import create_chat_completion


class ResearchConductor:
    """Manages and coordinates the research process.

    This class handles the main research workflow including planning
    research queries, conducting web searches, managing MCP retrievers,
    and gathering context from various sources.

    Attributes:
        researcher: The parent GPTResearcher instance.
        logger: Logger for research events.
        json_handler: Handler for JSON logging.
    """

    def __init__(self, researcher):
        """Initialize the ResearchConductor.

        Args:
            researcher: The GPTResearcher instance that owns this conductor.
        """
        self.researcher = researcher
        self.logger = logging.getLogger('research')
        self.json_handler = get_json_handler()
        # Add cache for MCP results to avoid redundant calls
        self._mcp_results_cache = None
        # Guards cache population when research passes run concurrently
        self._mcp_cache_lock = asyncio.Lock()
        # Track MCP query count for balanced mode
        self._mcp_query_count = 0
        self._pending_shared_urls: set[str] = set()
        self._owned_shared_urls: set[str] = set()
        self.last_retrieval_diagnostics: dict = {
            "candidate_count": 0,
            "selected_count": 0,
            "scraped_count": 0,
            "integrity_rejected_count": 0,
            "accepted_count": 0,
            "compression": {},
        }

    def _trace(self, event_type: str, data: dict) -> None:
        trace = getattr(self.researcher, "trace_event", None)
        if trace:
            trace(event_type, data)

    def _stage_timing(self, stage: str, duration_seconds: float) -> None:
        policy = getattr(self.researcher, "research_policy", None)
        estimates = (
            getattr(policy, "estimated_stage_seconds", {}) if policy else {}
        )
        self._trace(
            "stage_timing",
            {
                "stage": stage,
                "duration_seconds": round(duration_seconds, 3),
                "estimated_seconds": float(estimates.get(stage, 0.0)),
            },
        )

    async def plan_research(self, query, query_domains=None):
        """Gets the sub-queries from the query
        Args:
            query: original query
        Returns:
            List of queries
        """
        await stream_output(
            "logs",
            "planning_research",
            f"🌐 Browsing the web to learn more about the task: {query}...",
            self.researcher.websocket,
        )

        evidence_enabled = getattr(self.researcher.cfg, "planning_evidence_enabled", False)
        planning_result_count = (
            getattr(self.researcher.cfg, "planning_search_results", 8)
            if evidence_enabled
            else self.researcher.cfg.max_search_results_per_query
        )
        search_results = await get_search_results(
            query,
            self.researcher.retrievers[0],
            query_domains,
            researcher=self.researcher,
            max_results=planning_result_count,
        )
        self.logger.info(f"Initial search results obtained: {len(search_results)} results")
        if evidence_enabled:
            evidence = [
                source_card(result, index + 1)
                for index, result in enumerate(search_results)
                if source_url(result)
            ]
            if self.json_handler:
                self.json_handler.log_event("planning_evidence", {
                    "query": query,
                    "results": evidence,
                })
            self._trace("planning_evidence", {
                "query": query,
                "results": evidence,
            })
            await stream_output(
                "logs",
                "planning_evidence",
                f"🧭 Collected {len(evidence)} preliminary result cards for evidence-grounded planning.",
                self.researcher.websocket,
                True,
                evidence,
            )

        await stream_output(
            "logs",
            "planning_research",
            f"🤔 Planning the research strategy and subtasks...",
            self.researcher.websocket,
        )

        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        # Remove duplicate logging - this will be logged once in conduct_research instead

        outline = await plan_research_outline(
            query=query,
            search_results=search_results,
            agent_role_prompt=self.researcher.role,
            cfg=self.researcher.cfg,
            parent_query=self.researcher.parent_query,
            report_type=self.researcher.report_type,
            cost_callback=self.researcher.add_costs,
            retriever_names=retriever_names,  # Pass retriever names for MCP optimization
            **self.researcher.kwargs
        )
        self.logger.info(f"Research outline planned: {outline}")
        return outline

    async def conduct_research(self):
        """Runs the GPT Researcher to conduct research"""
        if self.json_handler:
            self.json_handler.update_content("query", self.researcher.query)
        
        self.logger.info(f"Starting research for query: {self.researcher.query}")
        
        # Log active retrievers once at the start of research
        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        self.logger.info(f"Active retrievers: {retriever_names}")
        
        # Note: visited_urls is deliberately NOT cleared here. It may be
        # shared with a parent researcher (e.g. detailed reports pass their
        # accumulated URLs into each subtopic researcher) so that already
        # scraped URLs are not fetched again.
        research_data = []

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "starting_research",
                f"🔍 Starting the research task for '{self.researcher.query}'...",
                self.researcher.websocket,
            )
            await stream_output(
                "logs",
                "agent_generated",
                self.researcher.agent,
                self.researcher.websocket
            )

        # Focused deep branches receive their parent role.  Avoid spending a
        # model request choosing a role for each deliberately narrow leaf.
        # Choose agent and role if not already defined.
        if not (self.researcher.agent and self.researcher.role):
            self.researcher.agent, self.researcher.role = await choose_agent(
                query=self.researcher.query,
                cfg=self.researcher.cfg,
                parent_query=self.researcher.parent_query,
                cost_callback=self.researcher.add_costs,
                headers=self.researcher.headers,
                prompt_family=self.researcher.prompt_family
            )
                
        # Check if MCP retrievers are configured
        has_mcp_retriever = any("mcpretriever" in r.__name__.lower() for r in self.researcher.retrievers)
        if has_mcp_retriever:
            self.logger.info("MCP retrievers configured and will be used with standard research flow")

        # Conduct research based on the source type
        if self.researcher.source_urls:
            self.logger.info("Using provided source URLs")
            research_data = await self._get_context_by_urls(self.researcher.source_urls)
            if research_data and len(research_data) == 0 and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "answering_from_memory",
                    f"🧐 I was unable to find relevant context in the provided sources...",
                    self.researcher.websocket,
                )
            if self.researcher.complement_source_urls:
                self.logger.info("Complementing with web search")
                additional_research = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
                research_data += ' '.join(additional_research)
        elif self.researcher.report_source == ReportSource.Web.value:
            self.logger.info("Using web search with all configured retrievers")
            research_data = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
        elif self.researcher.report_source == ReportSource.Local.value:
            self.logger.info("Using local search")
            document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            self.logger.info(f"Loaded {len(document_data)} documents")
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)

            research_data = await self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains)
        # Hybrid search including both local documents and web sources
        elif self.researcher.report_source == ReportSource.Hybrid.value:
            if self.researcher.document_urls:
                document_data = await OnlineDocumentLoader(self.researcher.document_urls).load()
            else:
                document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)
            # The local-docs pass and the web pass are independent, so run
            # them concurrently; visited_urls still dedupes across both.
            docs_context, web_context = await asyncio.gather(
                self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains),
                self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains),
            )
            research_data = self.researcher.prompt_family.join_local_web_documents(docs_context, web_context)
        elif self.researcher.report_source == ReportSource.Azure.value:
            from ..document.azure_document_loader import AzureDocumentLoader
            azure_loader = AzureDocumentLoader(
                container_name=os.getenv("AZURE_CONTAINER_NAME"),
                connection_string=os.getenv("AZURE_CONNECTION_STRING")
            )
            azure_files = await azure_loader.load()
            document_data = await DocumentLoader(azure_files).load()  # Reuse existing loader
            research_data = await self._get_context_by_web_search(self.researcher.query, document_data)

        elif self.researcher.report_source == ReportSource.LangChainDocuments.value:
            langchain_documents_data = await LangChainDocumentLoader(
                self.researcher.documents
            ).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(langchain_documents_data)
            research_data = await self._get_context_by_web_search(
                self.researcher.query, langchain_documents_data, self.researcher.query_domains
            )
        elif self.researcher.report_source == ReportSource.LangChainVectorStore.value:
            research_data = await self._get_context_by_vectorstore(self.researcher.query, self.researcher.vector_store_filter)

        # Rank and curate the sources
        self.researcher.context = research_data
        if self.researcher.cfg.curate_sources:
            self.logger.info("Curating sources")
            curated = await self.researcher.source_curator.curate_sources(research_data)
            # curate_sources() returns List[dict] with Title/Content/Source keys.
            # Normalize to str so downstream code that expects researcher.context
            # to be a string (e.g. "\n".join, .split(), len()) doesn't crash.
            if isinstance(curated, list):
                self.researcher.context = "\n\n".join(
                    "Title: {title}\nContent: {content}\nSource: {source}".format(
                        title=s.get("Title", ""),
                        content=s.get("Content", ""),
                        source=s.get("Source", ""),
                    ) if isinstance(s, dict) else str(s)
                    for s in curated
                )
            else:
                self.researcher.context = curated

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_step_finalized",
                f"Finalized research step.\n💸 Total Research Costs: ${self.researcher.get_costs()}",
                self.researcher.websocket,
            )
            if self.json_handler:
                self.json_handler.update_content("costs", self.researcher.get_costs())
                self.json_handler.update_content("context", self.researcher.context)

        self.logger.info(f"Research completed. Context size: {len(str(self.researcher.context))}")
        return self.researcher.context

    async def _get_context_by_urls(self, urls):
        """Scrapes and compresses the context from the given urls"""
        self.logger.info(f"Getting context from URLs: {urls}")
        
        new_search_urls = await self._get_new_urls(urls)
        shared_wait_urls = self._take_pending_shared_urls()
        self.logger.info(f"New URLs to process: {new_search_urls}")

        integrity_enabled = getattr(
            self.researcher.cfg, "post_scrape_source_integrity", False
        )
        scraped_content = await self.researcher.scraper_manager.browse_urls(
            new_search_urls, record_sources=not integrity_enabled
        )
        await self._publish_shared_scrapes(
            new_search_urls, scraped_content
        )
        scraped_content.extend(
            await self._collect_shared_scrapes(shared_wait_urls)
        )
        scrape_failures = list(
            getattr(
                self.researcher.scraper_manager,
                "last_scrape_failures",
                [],
            )
        )

        # A selected canonical page may fail while its raw, HTML, or PDF form
        # remains fetchable. Try only deterministic same-source variants.
        selected_candidates_by_url = {
            url: {"url": url} for url in new_search_urls
        }
        fetched_urls = {source_url(item) for item in scraped_content}
        alternate_candidates: dict[str, dict] = {}
        for url in new_search_urls:
            if url in fetched_urls:
                continue
            original = selected_candidates_by_url.get(url, {"url": url})
            for alternate in canonical_source_alternatives(url):
                alternate_candidates[alternate] = {
                    **original,
                    "url": alternate,
                    "href": alternate,
                }
        alternate_urls = await self._get_new_urls(list(alternate_candidates))
        alternate_wait_urls = self._take_pending_shared_urls()
        if alternate_urls:
            selected_candidates_by_url.update(
                {
                    url: alternate_candidates[url]
                    for url in alternate_urls
                    if url in alternate_candidates
                }
            )
            alternate_content = await self.researcher.scraper_manager.browse_urls(
                alternate_urls, record_sources=not integrity_enabled
            )
            await self._publish_shared_scrapes(
                alternate_urls, alternate_content
            )
            scrape_failures.extend(
                getattr(
                    self.researcher.scraper_manager,
                    "last_scrape_failures",
                    [],
                )
            )
            scraped_content.extend(alternate_content)
            self.last_retrieval_diagnostics["canonical_alternative_attempt_count"] = (
                len(alternate_urls)
            )
            self.last_retrieval_diagnostics["canonical_alternative_success_count"] = (
                len(alternate_content)
            )
            self._trace(
                "canonical_source_recovery",
                {
                    "query": self.researcher.query,
                    "attempted_urls": alternate_urls,
                    "accepted_before_integrity": len(alternate_content),
                },
            )
        scraped_content.extend(
            await self._collect_shared_scrapes(alternate_wait_urls)
        )
        if integrity_enabled:
            scraped_content = await self._validate_post_scrape_sources(
                self.researcher.query,
                scraped_content,
                selected_candidates_by_url,
            )
        self.logger.info(f"Scraped content from {len(scraped_content)} URLs")
        self._trace("direct_url_retrieval", {
            "requested_urls": list(urls),
            "new_urls": new_search_urls,
            "accepted_count": len(scraped_content),
        })

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        compression_started_at = time.perf_counter()
        compression = (
            await self.researcher.context_manager.get_similar_content_with_diagnostics(
                self.researcher.query, scraped_content
            )
        )
        self.last_retrieval_diagnostics.update(
            {
                "candidate_count": len(urls),
                "selected_count": len(new_search_urls),
                "scraped_count": len(scraped_content),
                "accepted_count": len(scraped_content),
                "scrape_failures": scrape_failures,
                "compression": compression.diagnostics(),
                "compression_duration_seconds": round(
                    time.perf_counter() - compression_started_at, 3
                ),
            }
        )
        if scrape_failures:
            self._trace(
                "scrape_failures",
                {
                    "query": self.researcher.query,
                    "failures": scrape_failures,
                },
            )
        self._trace(
            "compression",
            {
                "query": self.researcher.query,
                **compression.diagnostics(),
            },
        )
        self._stage_timing(
            "compression", time.perf_counter() - compression_started_at
        )
        return compression.context

    # Add logging to other methods similarly...

    async def _get_context_by_vectorstore(self, query, filter: dict | None = None):
        """
        Generates the context for the research task by searching the vectorstore
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting vectorstore search for query: {query}")
        context = []
        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query)
        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️  I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # Using asyncio.gather to process the sub_queries asynchronously
        context = await asyncio.gather(
            *[
                self._process_sub_query_with_vectorstore(sub_query, filter)
                for sub_query in sub_queries
            ]
        )
        return context

    async def _get_context_by_web_search(self, query, scraped_data: list | None = None, query_domains: list | None = None):
        """
        Generates the context for the research task by searching the query and scraping the results
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting web search for query: {query}")
        
        if scraped_data is None:
            scraped_data = []
        if query_domains is None:
            query_domains = []

        # **CONFIGURABLE MCP OPTIMIZATION: Control MCP strategy**
        mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
        
        # Get MCP strategy configuration
        mcp_strategy = self._get_mcp_strategy()
        
        # Lock so concurrent research passes (e.g. hybrid mode) populate the
        # MCP cache once instead of racing to run the same MCP research twice.
        async with self._mcp_cache_lock:
            if mcp_retrievers and self._mcp_results_cache is None:
                if mcp_strategy == "disabled":
                    # MCP disabled - skip MCP research entirely
                    self.logger.info("MCP disabled by strategy, skipping MCP research")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_disabled",
                            f"⚡ MCP research disabled by configuration",
                            self.researcher.websocket,
                        )
                elif mcp_strategy == "fast":
                    # Fast: Run MCP once with original query
                    self.logger.info("MCP fast strategy: Running once with original query")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_optimization",
                            f"🚀 MCP Fast: Running once for main query (performance mode)",
                            self.researcher.websocket,
                        )

                    # Execute MCP research once with the original query
                    mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                    self._mcp_results_cache = mcp_context
                    self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")
                elif mcp_strategy == "deep":
                    # Deep: Will run MCP for all queries (original behavior) - defer to per-query execution
                    self.logger.info("MCP deep strategy: Will run for all queries")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_comprehensive",
                            f"🔍 MCP Deep: Will run for each sub-query (thorough mode)",
                            self.researcher.websocket,
                        )
                    # Don't cache - let each sub-query run MCP individually
                else:
                    # Unknown strategy - default to fast
                    self.logger.warning(f"Unknown MCP strategy '{mcp_strategy}', defaulting to fast")
                    mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                    self._mcp_results_cache = mcp_context
                    self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")

        focused = getattr(self.researcher, "research_mode", "standard") == "deep_branch"
        if focused:
            sub_queries = [query]
            self.logger.info("Focused deep branch: skipping nested query planning for %r", query)
        else:
            # Generate Sub-Queries including original query
            sub_queries = await self.plan_research(query, query_domains)
            self.logger.info(f"Generated sub-queries: {sub_queries}")

            # If this is not part of a sub researcher, add original query to research for better results
            if self.researcher.report_type != "subtopic_report":
                sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️ I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # Using asyncio.gather to process the sub_queries asynchronously
        try:
            context = await asyncio.gather(
                *[
                    self._process_sub_query(sub_query, scraped_data, query_domains)
                    for sub_query in sub_queries
                ]
            )
            self.logger.info(f"Gathered context from {len(context)} sub-queries")
            # Filter out empty results and join the context
            context = [c for c in context if c]
            if context:
                combined_context = " ".join(context)
                self.logger.info(f"Combined context size: {len(combined_context)}")
                return combined_context
            return []
        except Exception as e:
            self.logger.error(f"Error during web search: {e}", exc_info=True)
            return []

    def _get_mcp_strategy(self) -> str:
        """
        Get the MCP strategy configuration.
        
        Priority:
        1. Instance-level setting (self.researcher.mcp_strategy)
        2. Config file setting (self.researcher.cfg.mcp_strategy) 
        3. Default value ("fast")
        
        Returns:
            str: MCP strategy
                "disabled" = Skip MCP entirely
                "fast" = Run MCP once with original query (default)
                "deep" = Run MCP for all sub-queries
        """
        # Check instance-level setting first
        if hasattr(self.researcher, 'mcp_strategy') and self.researcher.mcp_strategy is not None:
            return self.researcher.mcp_strategy
        
        # Check config setting
        if hasattr(self.researcher.cfg, 'mcp_strategy'):
            return self.researcher.cfg.mcp_strategy
        
        # Default to fast mode
        return "fast"

    async def _execute_mcp_research_for_queries(self, queries: list, mcp_retrievers: list) -> list:
        """
        Execute MCP research for a list of queries.
        
        Args:
            queries: List of queries to research
            mcp_retrievers: List of MCP retriever classes
            
        Returns:
            list: Combined MCP context entries from all queries
        """
        all_mcp_context = []
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"Executing MCP research for query {i}/{len(queries)}: {query}")
            
            for retriever in mcp_retrievers:
                try:
                    mcp_results = await self._execute_mcp_research(retriever, query)
                    if mcp_results:
                        for result in mcp_results:
                            content = result.get("body", "")
                            url = result.get("href", "")
                            title = result.get("title", "")
                            
                            if content:
                                context_entry = {
                                    "content": content,
                                    "url": url,
                                    "title": title,
                                    "query": query,
                                    "source_type": "mcp"
                                }
                                all_mcp_context.append(context_entry)
                        
                        self.logger.info(f"Added {len(mcp_results)} MCP results for query: {query}")
                        
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results_cached",
                                f"✅ Cached {len(mcp_results)} MCP results from query {i}/{len(queries)}",
                                self.researcher.websocket,
                            )
                except Exception as e:
                    self.logger.error(f"Error in MCP research for query '{query}': {e}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_error",
                            f"⚠️ MCP research error for query {i}, continuing with other sources",
                            self.researcher.websocket,
                        )
        
        return all_mcp_context

    def _tavily_mcp_redundant_with_direct(self, mcp_retrievers, non_mcp_retrievers) -> bool:
        """True when MCP would only re-query Tavily while direct Tavily is active.

        The frontend Tavily Web Search MCP preset hits the same API as
        `TavilySearch` and adds extra LLM tool-selection cost for no new data
        when both run together (#1875).
        """
        if not mcp_retrievers or not non_mcp_retrievers:
            return False
        has_direct_tavily = any(
            getattr(r, "__name__", "").lower() == "tavilysearch" for r in non_mcp_retrievers
        )
        if not has_direct_tavily:
            return False
        configs = getattr(self.researcher, "mcp_configs", None) or []
        if not configs:
            return False
        # If every configured MCP server is a Tavily MCP package, treat as redundant.
        def _is_tavily_mcp(cfg: dict) -> bool:
            name = str(cfg.get("name", "")).lower()
            args = " ".join(str(a) for a in (cfg.get("args") or [])).lower()
            command = str(cfg.get("command", "")).lower()
            blob = f"{name} {args} {command}"
            return "tavily" in blob

        return all(isinstance(c, dict) and _is_tavily_mcp(c) for c in configs)


    async def _process_sub_query(self, sub_query: str, scraped_data: list = [], query_domains: list = []):
        """Takes in a sub query and scrapes urls based on it and gathers context."""
        if self.json_handler:
            self.json_handler.log_event("sub_query", {
                "query": sub_query,
                "scraped_data_size": len(scraped_data)
            })
        self._trace("sub_query", {
            "query": sub_query,
            "scraped_data_size": len(scraped_data),
        })
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        try:
            # Identify MCP retrievers
            mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
            non_mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" not in r.__name__.lower()]

            # Avoid dual Tavily path (direct retriever + tavily-mcp) under default RETRIEVER=tavily.
            if self._tavily_mcp_redundant_with_direct(mcp_retrievers, non_mcp_retrievers):
                self.logger.warning(
                    "Skipping LLM MCP Tavily path because TavilySearch is already configured as a direct retriever; set RETRIEVER without tavily or use non-Tavily MCP servers to keep MCP."
                )
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_tavily_deduped",
                        "⚠️ Skipping Tavily MCP (redundant with direct Tavily retriever) to avoid double API cost",
                        self.researcher.websocket,
                    )
                mcp_retrievers = []
            
            # Initialize context components
            mcp_context = []
            web_context = ""
            
            # Get MCP strategy configuration
            mcp_strategy = self._get_mcp_strategy()
            
            # **CONFIGURABLE MCP PROCESSING**
            if mcp_retrievers:
                if mcp_strategy == "disabled":
                    # MCP disabled - skip entirely
                    self.logger.info(f"MCP disabled for sub-query: {sub_query}")
                elif mcp_strategy == "fast" and self._mcp_results_cache is not None:
                    # Fast: Use cached results
                    mcp_context = self._mcp_results_cache.copy()
                    
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_reuse",
                            f"♻️ Reusing cached MCP results ({len(mcp_context)} sources) for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    self.logger.info(f"Reused {len(mcp_context)} cached MCP results for sub-query: {sub_query}")
                elif mcp_strategy == "deep":
                    # Deep: Run MCP for every sub-query
                    self.logger.info(f"Running deep MCP research for: {sub_query}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_comprehensive_run",
                            f"🔍 Running deep MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
                else:
                    # Fallback: if no cache and not deep mode, run MCP for this query
                    self.logger.warning("MCP cache not available, falling back to per-sub-query execution")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_fallback",
                            f"🔌 MCP cache unavailable, running MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
            
            # Get web search context using non-MCP retrievers (if no scraped data provided)
            if not scraped_data:
                scraped_data = await self._scrape_data_by_urls(sub_query, query_domains)
                self.logger.info(f"Scraped data size: {len(scraped_data)}")

            # Get similar content based on scraped data
            if scraped_data:
                compression_started_at = time.perf_counter()
                compression = (
                    await self.researcher.context_manager.get_similar_content_with_diagnostics(
                        sub_query, scraped_data
                    )
                )
                web_context = compression.context
                self.last_retrieval_diagnostics["compression"] = (
                    compression.diagnostics()
                )
                self.last_retrieval_diagnostics["compression_duration_seconds"] = (
                    round(time.perf_counter() - compression_started_at, 3)
                )
                self._stage_timing(
                    "compression",
                    time.perf_counter() - compression_started_at,
                )
                self._trace(
                    "compression",
                    {
                        "query": sub_query,
                        **compression.diagnostics(),
                    },
                )
                self.logger.info(f"Web content found for sub-query: {len(str(web_context)) if web_context else 0} chars")

            # Combine MCP context with web context intelligently
            combined_context = self._combine_mcp_and_web_context(mcp_context, web_context, sub_query)
            
            # Log context combination results
            if combined_context:
                context_length = len(str(combined_context))
                self.logger.info(f"Combined context for '{sub_query}': {context_length} chars")
                
                if self.researcher.verbose:
                    mcp_count = len(mcp_context)
                    web_available = bool(web_context)
                    cache_used = self._mcp_results_cache is not None and mcp_retrievers and mcp_strategy != "deep"
                    cache_status = " (cached)" if cache_used else ""
                    await stream_output(
                        "logs",
                        "context_combined",
                        f"📚 Combined research context: {mcp_count} MCP sources{cache_status}, {'web content' if web_available else 'no web content'}",
                        self.researcher.websocket,
                    )
            else:
                self.logger.warning(f"No combined context found for sub-query: {sub_query}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "subquery_context_not_found",
                        f"🤷 No content found for '{sub_query}'...",
                        self.researcher.websocket,
                    )
            
            if combined_context and self.json_handler:
                self.json_handler.log_event("content_found", {
                    "sub_query": sub_query,
                    "content_size": len(str(combined_context)),
                    "mcp_sources": len(mcp_context),
                    "web_content": bool(web_context)
                })
            if combined_context:
                self._trace("content_found", {
                    "sub_query": sub_query,
                    "content_size": len(str(combined_context)),
                    "mcp_sources": len(mcp_context),
                    "web_content": bool(web_context),
                })
                
            return combined_context
            
        except Exception as e:
            self.logger.error(f"Error processing sub-query {sub_query}: {e}", exc_info=True)
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subquery_error",
                    f"❌ Error processing '{sub_query}': {str(e)}",
                    self.researcher.websocket,
                )
            return ""

    async def _execute_mcp_research(self, retriever, query):
        """
        Execute MCP research using the new two-stage approach.
        
        Args:
            retriever: The MCP retriever class
            query: The search query
            
        Returns:
            list: MCP research results
        """
        retriever_name = retriever.__name__
        
        self.logger.info(f"Executing MCP research with {retriever_name} for query: {query}")
        
        try:
            # Instantiate the MCP retriever with proper parameters
            # Pass the researcher instance (self.researcher) which contains both cfg and mcp_configs
            retriever_instance = retriever(
                query=query, 
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket,
                researcher=self.researcher  # Pass the entire researcher instance
            )
            
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval_stage1",
                    f"🧠 Stage 1: Selecting optimal MCP tools for: {query}",
                    self.researcher.websocket,
                )
            
            # Execute the two-stage MCP search
            results = retriever_instance.search(
                max_results=self.researcher.cfg.max_search_results_per_query
            )
            
            if results:
                result_count = len(results)
                self.logger.info(f"MCP research completed: {result_count} results from {retriever_name}")
                
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_research_complete",
                        f"🎯 MCP research completed: {result_count} intelligent results obtained",
                        self.researcher.websocket,
                    )
                
                return results
            else:
                self.logger.info(f"No results returned from MCP research with {retriever_name}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_no_results",
                        f"ℹ️ No relevant information found via MCP for: {query}",
                        self.researcher.websocket,
                    )
                return []
                
        except Exception as e:
            self.logger.error(f"Error in MCP research with {retriever_name}: {str(e)}")
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_research_error",
                    f"⚠️ MCP research error: {str(e)} - continuing with other sources",
                    self.researcher.websocket,
                )
            return []

    def _combine_mcp_and_web_context(self, mcp_context: list, web_context: str, sub_query: str) -> str:
        """
        Intelligently combine MCP and web research context.
        
        Args:
            mcp_context: List of MCP context entries
            web_context: Web research context string  
            sub_query: The sub-query being processed
            
        Returns:
            str: Combined context string
        """
        combined_parts = []
        
        # Add web context first if available
        if web_context and web_context.strip():
            combined_parts.append(web_context.strip())
            self.logger.debug(f"Added web context: {len(web_context)} chars")
        
        # Add MCP context with proper formatting
        if mcp_context:
            mcp_formatted = []
            
            for i, item in enumerate(mcp_context):
                content = item.get("content", "")
                url = item.get("url", "")
                title = item.get("title", f"MCP Result {i+1}")
                
                if content and content.strip():
                    # Create a well-formatted context entry
                    if url and url != f"mcp://llm_analysis":
                        citation = f"\n\n*Source: {title} ({url})*"
                    else:
                        citation = f"\n\n*Source: {title}*"
                    
                    formatted_content = f"{content.strip()}{citation}"
                    mcp_formatted.append(formatted_content)
            
            if mcp_formatted:
                # Join MCP results with clear separation
                mcp_section = "\n\n---\n\n".join(mcp_formatted)
                combined_parts.append(mcp_section)
                self.logger.debug(f"Added {len(mcp_context)} MCP context entries")
        
        # Combine all parts
        if combined_parts:
            final_context = "\n\n".join(combined_parts)
            self.logger.info(f"Combined context for '{sub_query}': {len(final_context)} total chars")
            return final_context
        else:
            self.logger.warning(f"No context to combine for sub-query: {sub_query}")
            return ""

    async def _process_sub_query_with_vectorstore(self, sub_query: str, filter: dict | None = None):
        """Takes in a sub query and gathers context from the user provided vector store

        Args:
            sub_query (str): The sub-query generated from the original query

        Returns:
            str: The context gathered from search
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_with_vectorstore_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        context = await self.researcher.context_manager.get_similar_content_by_query_with_vectorstore(sub_query, filter)

        return context

    async def _publish_shared_scrapes(
        self,
        requested_urls: list[str],
        scraped_content: list[dict],
    ) -> None:
        cache = getattr(self.researcher, "shared_scrape_cache", None)
        futures = getattr(self.researcher, "shared_scrape_futures", None)
        if cache is None or futures is None:
            return
        by_url = {
            source_url(item): dict(item)
            for item in scraped_content
            if source_url(item)
        }
        async with self.researcher.visited_urls_lock:
            for url in requested_urls:
                value = by_url.get(url)
                cache[url] = value
                future = futures.get(url)
                if future is not None and not future.done():
                    future.set_result(value)
                self._owned_shared_urls.discard(url)

    async def _collect_shared_scrapes(
        self,
        urls: set[str],
    ) -> list[dict]:
        cache = getattr(self.researcher, "shared_scrape_cache", None)
        futures = getattr(self.researcher, "shared_scrape_futures", None)
        if not urls or cache is None or futures is None:
            return []
        values: list[dict] = []
        for url in sorted(urls):
            value = cache.get(url)
            if value is None:
                future = futures.get(url)
                if future is not None:
                    value = await asyncio.shield(future)
            if value:
                values.append(dict(value))
        if values:
            self._trace(
                "shared_scrape_cache",
                {
                    "requested_count": len(urls),
                    "hit_count": len(values),
                    "urls": sorted(
                        source_url(item) for item in values if source_url(item)
                    ),
                },
            )
        return values

    def _take_pending_shared_urls(self) -> set[str]:
        pending = set(self._pending_shared_urls)
        self._pending_shared_urls.clear()
        return pending

    async def _get_new_urls(self, url_set_input):
        """Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """

        new_urls = []
        lock = getattr(self.researcher, "visited_urls_lock", None)
        if lock is None:
            lock = asyncio.Lock()
        async with lock:
            for url in url_set_input:
                if url not in self.researcher.visited_urls:
                    self.researcher.visited_urls.add(url)
                    new_urls.append(url)
                    futures = getattr(
                        self.researcher, "shared_scrape_futures", None
                    )
                    if futures is not None:
                        future = futures.get(url)
                        if future is None:
                            future = asyncio.get_running_loop().create_future()
                            futures[url] = future
                        self._owned_shared_urls.add(url)
                elif url not in self._owned_shared_urls:
                    cache = getattr(
                        self.researcher, "shared_scrape_cache", None
                    )
                    futures = getattr(
                        self.researcher, "shared_scrape_futures", None
                    )
                    if (
                        cache is not None
                        and futures is not None
                        and (url in cache or url in futures)
                    ):
                        self._pending_shared_urls.add(url)
        for url in new_urls:
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "added_source_url",
                    f"✅ Added source url to research: {url}\n",
                    self.researcher.websocket,
                    True,
                    url,
                )

        return new_urls

    async def _search_relevant_source_urls(self, query, query_domains: list | None = None):
        search_started_at = time.perf_counter()
        candidates = []
        if query_domains is None:
            query_domains = []

        # Iterate through the currently set retrievers
        # This allows the method to work when retrievers are temporarily modified
        for retriever_class in self.researcher.retrievers:
            # Skip MCP retrievers as they don't provide URLs for scraping
            if "mcpretriever" in retriever_class.__name__.lower():
                continue

            try:
                # Instantiate the retriever with the sub-query
                retriever = retriever_class(query, query_domains=query_domains)

                # Perform the search using the current retriever
                policy = getattr(self.researcher, "research_policy", None)
                max_results = (
                    policy.result_cards_per_query
                    if policy is not None
                    else self.researcher.cfg.max_search_results_per_query
                )
                search_results = await asyncio.to_thread(
                    retriever.search, max_results=max_results
                )

                if not search_results:
                    continue

                # Retain result metadata until source selection has made a decision.
                for result in search_results:
                    url = result.get("href") or result.get("url")
                    if url:
                        candidates.append(result)
            except Exception as e:
                self.logger.error(f"Error searching with {retriever_class.__name__}: {e}")

        # Preserve first-seen ordering while suppressing duplicate result cards.
        deduplicated = []
        seen_urls = set()
        for candidate in candidates:
            url = source_url(candidate)
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduplicated.append(candidate)
        self._trace("search_results", {
            "query": query,
            "candidate_count": len(candidates),
            "deduplicated_count": len(deduplicated),
            "candidates": [
                source_card(candidate, index + 1)
                for index, candidate in enumerate(deduplicated)
            ],
        })
        self.last_retrieval_diagnostics["candidate_count"] = len(deduplicated)
        self.last_retrieval_diagnostics["search_duration_seconds"] = round(
            time.perf_counter() - search_started_at, 3
        )
        self._stage_timing("search", time.perf_counter() - search_started_at)

        selected = deduplicated
        if getattr(self.researcher.cfg, "pre_scrape_source_curation", False) and deduplicated:
            selected = await self._select_source_candidates(query, deduplicated)
        self.last_retrieval_diagnostics["selected_count"] = len(selected)

        new_search_urls = []
        prefetched_content = []
        selected_candidates_by_url = {}
        for result in selected:
            url = source_url(result)
            selected_candidates_by_url[url] = result
            raw_content = result.get("raw_content")
            if raw_content and len(raw_content) > 100:
                prefetched = {
                    "url": url,
                    "raw_content": raw_content,
                    "title": result.get("title", ""),
                }
                if result.get("_gptr_source_tier"):
                    prefetched["_gptr_source_tier"] = result[
                        "_gptr_source_tier"
                    ]
                prefetched_content.append(prefetched)
            else:
                new_search_urls.append(url)

        new_search_urls = await self._get_new_urls(new_search_urls)

        return new_search_urls, prefetched_content, selected_candidates_by_url

    async def _select_source_candidates(self, query, candidates):
        """Use the fast model to choose a small, diverse pre-scrape set.

        Invalid output and transient model failures intentionally use the same
        deterministic fallback so the crawler never falls back to scraping every
        candidate result.
        """
        policy = getattr(self.researcher, "research_policy", None)
        max_sources = max(
            1,
            (
                policy.scrape_cap_per_query
                if policy is not None
                else getattr(
                    self.researcher.cfg,
                    "pre_scrape_max_sources_per_query",
                    3,
                )
            ),
        )
        strict = (
            getattr(self.researcher, "research_mode", "standard").startswith("deep_branch")
            and getattr(self.researcher.cfg, "deep_research_source_standards", False)
        )
        mode = str(getattr(self.researcher.cfg, "source_selector_mode", "llm")).lower()
        cards = [source_card(candidate, index + 1) for index, candidate in enumerate(candidates)]
        selection_started_at = time.perf_counter()
        used_fallback = False
        selector_mode = "deterministic"
        parsed = None
        if should_use_model_selector(query, candidates, mode):
            selector_mode = "llm"
            try:
                from ..prompts import PromptFamily

                response = await create_chat_completion(
                    model=self.researcher.cfg.fast_llm_model,
                    messages=[{
                        "role": "user",
                        "content": PromptFamily.select_search_sources_prompt(query, cards, max_sources),
                    }],
                    temperature=0.1,
                    max_tokens=900,
                    llm_provider=self.researcher.cfg.fast_llm_provider,
                    llm_kwargs=self.researcher.cfg.llm_kwargs,
                    cost_callback=getattr(self.researcher, "add_costs", None),
                    **getattr(self.researcher, "kwargs", {}),
                )
                parsed = parse_model_selection(json_repair.loads(response), candidates, max_sources)
            except Exception as error:
                self.logger.warning("Pre-scrape source selection failed: %s", error)
                parsed = None
        if parsed is None:
            used_fallback = True
            selected, reasons = deterministic_select_sources(query, candidates, max_sources, strict=strict)
        else:
            selected, reasons = parsed
            has_higher_tier = any(
                source_quality_tier(candidate, query) in {"primary", "reputable"}
                and has_meaningful_query_anchor(query, candidate)
                for candidate in candidates
            )
            off_topic = [
                candidate for candidate in selected
                if (
                    not has_meaningful_query_anchor(query, candidate)
                    or source_quality_tier(candidate, query) == "reject"
                    or (
                        strict
                        and source_quality_tier(candidate, query) == "fallback"
                        and has_higher_tier
                    )
                )
            ]
            if off_topic:
                selected = [candidate for candidate in selected if candidate not in off_topic]
                for candidate in off_topic:
                    reasons[source_url(candidate)] = "rejected by deep source-standard or query-anchor guard"
                if not selected:
                    used_fallback = True
                    selected, fallback_reasons = deterministic_select_sources(query, candidates, max_sources, strict=strict)
                    reasons.update(fallback_reasons)

        for candidate in selected:
            candidate["_gptr_source_tier"] = source_quality_tier(
                candidate, query
            )
        selected_urls = {source_url(candidate) for candidate in selected}
        selection_event = {
            "query": query,
            "policy": getattr(self.researcher.cfg, "source_curation_policy", "balanced"),
            "selector_mode": selector_mode,
            "strict_standards": strict,
            "fallback": used_fallback,
            "duration_ms": round((time.perf_counter() - selection_started_at) * 1000, 1),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "candidates": cards,
            "selected": [
                {"url": source_url(candidate), "tier": source_quality_tier(candidate, query), "reason": reasons.get(source_url(candidate), "selected")}
                for candidate in selected
            ],
            "selected_urls": [source_url(candidate) for candidate in selected],
            "rejected": [
                {"url": source_url(candidate), "tier": source_quality_tier(candidate, query), "reason": reasons.get(source_url(candidate), "not selected")}
                for candidate in candidates
                if source_url(candidate) not in selected_urls
            ],
        }
        self.last_retrieval_diagnostics["selector_mode"] = selector_mode
        self.last_retrieval_diagnostics["selection_duration_seconds"] = round(
            time.perf_counter() - selection_started_at, 3
        )
        self._stage_timing(
            "selection", time.perf_counter() - selection_started_at
        )
        if self.json_handler:
            self.json_handler.log_event("source_selection", selection_event)
        self._trace("source_selection", selection_event)
        await stream_output(
            "logs",
            "source_selection",
            f"🏅 Selected {len(selected)}/{len(candidates)} balanced web sources before scraping.",
            self.researcher.websocket,
            True,
            selection_event,
        )
        return selected

    async def _scrape_data_by_urls(self, sub_query, query_domains: list | None = None):
        """
        Runs a sub-query across multiple retrievers and scrapes the resulting URLs.
        Retrievers that already provide full content (e.g. PubMed Central) have their
        content passed through directly without re-scraping.

        Args:
            sub_query (str): The sub-query to search for.

        Returns:
            list: A list of scraped content results.
        """
        if query_domains is None:
            query_domains = []

        new_search_urls, prefetched_content, selected_candidates_by_url = await self._search_relevant_source_urls(
            sub_query, query_domains
        )
        shared_wait_urls = self._take_pending_shared_urls()
        scrape_stage_started_at = time.perf_counter()

        # Log the research process if verbose mode is on
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "researching",
                f"🤔 Researching for relevant information across multiple sources...\n",
                self.researcher.websocket,
            )

        # Scrape URLs that need fetching (skip those already provided by retrievers)
        integrity_enabled = getattr(self.researcher.cfg, "post_scrape_source_integrity", False)
        scraped_content = await self.researcher.scraper_manager.browse_urls(
            new_search_urls, record_sources=not integrity_enabled
        )
        await self._publish_shared_scrapes(
            new_search_urls, scraped_content
        )
        scraped_content.extend(
            await self._collect_shared_scrapes(shared_wait_urls)
        )
        scrape_failures = list(
            getattr(
                self.researcher.scraper_manager,
                "last_scrape_failures",
                [],
            )
        )

        fetched_urls = {source_url(item) for item in scraped_content}
        alternate_candidates: dict[str, dict] = {}
        for url in new_search_urls:
            if url in fetched_urls:
                continue
            original = selected_candidates_by_url.get(url, {"url": url})
            for alternate in canonical_source_alternatives(url):
                alternate_candidates[alternate] = {
                    **original,
                    "url": alternate,
                    "href": alternate,
                }
        alternate_urls = await self._get_new_urls(list(alternate_candidates))
        alternate_wait_urls = self._take_pending_shared_urls()
        if alternate_urls:
            selected_candidates_by_url.update(
                {
                    url: alternate_candidates[url]
                    for url in alternate_urls
                    if url in alternate_candidates
                }
            )
            alternate_content = await self.researcher.scraper_manager.browse_urls(
                alternate_urls, record_sources=not integrity_enabled
            )
            await self._publish_shared_scrapes(
                alternate_urls, alternate_content
            )
            scrape_failures.extend(
                getattr(
                    self.researcher.scraper_manager,
                    "last_scrape_failures",
                    [],
                )
            )
            scraped_content.extend(alternate_content)
            self.last_retrieval_diagnostics[
                "canonical_alternative_attempt_count"
            ] = len(alternate_urls)
            self.last_retrieval_diagnostics[
                "canonical_alternative_success_count"
            ] = len(alternate_content)
            self._trace(
                "canonical_source_recovery",
                {
                    "query": sub_query,
                    "attempted_urls": alternate_urls,
                    "accepted_before_integrity": len(alternate_content),
                },
            )
        scraped_content.extend(
            await self._collect_shared_scrapes(alternate_wait_urls)
        )

        # Merge pre-fetched content from retrievers that already provide full text
        scraped_content.extend(prefetched_content)
        self.last_retrieval_diagnostics["scraped_count"] = len(scraped_content)
        self.last_retrieval_diagnostics["scrape_failures"] = scrape_failures
        if scrape_failures:
            self._trace(
                "scrape_failures",
                {
                    "query": sub_query,
                    "failures": scrape_failures,
                },
            )

        if integrity_enabled:
            scraped_content = await self._validate_post_scrape_sources(
                sub_query, scraped_content, selected_candidates_by_url
            )
        self.last_retrieval_diagnostics["accepted_count"] = len(scraped_content)
        self.last_retrieval_diagnostics["scrape_duration_seconds"] = round(
            time.perf_counter() - scrape_stage_started_at, 3
        )
        self._stage_timing(
            "scraping", time.perf_counter() - scrape_stage_started_at
        )

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        return scraped_content

    async def _validate_post_scrape_sources(self, query, scraped_content, selected_candidates_by_url):
        """Keep only fetched pages that still match the selected evidence card."""
        accepted = []
        rejected = []
        for item in scraped_content:
            url = source_url(item)
            reason = post_scrape_integrity_reason(
                query, selected_candidates_by_url.get(url), item
            )
            if reason:
                rejected.append({"url": url, "reason": reason})
            else:
                selected_candidate = selected_candidates_by_url.get(url) or {}
                if selected_candidate.get("_gptr_source_tier"):
                    item["_gptr_source_tier"] = selected_candidate[
                        "_gptr_source_tier"
                    ]
                accepted.append(item)

        # Unlike the legacy browser path, only verified pages are exposed in
        # get_research_sources() and therefore to the final report writer.
        self.researcher.add_research_sources(accepted)
        event = {
            "query": query,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "rejected": rejected,
        }
        self.last_retrieval_diagnostics["integrity_rejected_count"] = len(
            rejected
        )
        if self.json_handler:
            self.json_handler.log_event("post_scrape_source_integrity", event)
        self._trace("post_scrape_source_integrity", event)
        if rejected:
            self.logger.info(
                "Post-scrape integrity accepted %s/%s pages for query %r",
                len(accepted), len(scraped_content), query,
            )
        await stream_output(
            "logs",
            "post_scrape_source_integrity",
            f"🛡️ Verified {len(accepted)}/{len(scraped_content)} fetched pages before using them as evidence.",
            self.researcher.websocket,
            True,
            event,
        )
        return accepted

    async def _search(self, retriever, query):
        """
        Perform a search using the specified retriever.
        
        Args:
            retriever: The retriever class to use
            query: The search query
            
        Returns:
            list: Search results
        """
        retriever_name = retriever.__name__
        is_mcp_retriever = "mcpretriever" in retriever_name.lower()
        
        self.logger.info(f"Searching with {retriever_name} for query: {query}")
        
        try:
            # Instantiate the retriever
            retriever_instance = retriever(
                query=query, 
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket if is_mcp_retriever else None,
                researcher=self.researcher if is_mcp_retriever else None
            )
            
            # Log MCP server configurations if using MCP retriever
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval",
                    f"🔌 Consulting MCP server(s) for information on: {query}",
                    self.researcher.websocket,
                )
            
            # Perform the search
            if hasattr(retriever_instance, 'search'):
                results = retriever_instance.search(
                    max_results=self.researcher.cfg.max_search_results_per_query
                )
                
                # Log result information
                if results:
                    result_count = len(results)
                    self.logger.info(f"Received {result_count} results from {retriever_name}")
                    
                    # Special logging for MCP retriever
                    if is_mcp_retriever:
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results",
                                f"✓ Retrieved {result_count} results from MCP server",
                                self.researcher.websocket,
                            )
                        
                        # Log result details
                        for i, result in enumerate(results[:3]):  # Log first 3 results
                            title = result.get("title", "No title")
                            url = result.get("href", "No URL")
                            content_length = len(result.get("body", "")) if result.get("body") else 0
                            self.logger.info(f"MCP result {i+1}: '{title}' from {url} ({content_length} chars)")
                            
                        if result_count > 3:
                            self.logger.info(f"... and {result_count - 3} more MCP results")
                else:
                    self.logger.info(f"No results returned from {retriever_name}")
                    if is_mcp_retriever and self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_no_results",
                            f"ℹ️ No relevant information found from MCP server for: {query}",
                            self.researcher.websocket,
                        )
                
                return results
            else:
                self.logger.error(f"Retriever {retriever_name} does not have a search method")
                return []
        except Exception as e:
            self.logger.error(f"Error searching with {retriever_name}: {str(e)}")
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_error",
                    f"❌ Error retrieving information from MCP server: {str(e)}",
                    self.researcher.websocket,
                )
            return []
            
    async def _extract_content(self, results):
        """
        Extract content from search results using the browser manager.
        
        Args:
            results: Search results
            
        Returns:
            list: Extracted content
        """
        self.logger.info(f"Extracting content from {len(results)} search results")
        
        # Get the URLs from the search results
        urls = []
        for result in results:
            if isinstance(result, dict) and "href" in result:
                urls.append(result["href"])
        
        # Skip if no URLs found
        if not urls:
            return []
            
        # Reserve new URLs under the tree-wide lock before scraping so
        # concurrent branches cannot fetch the same page.
        new_urls = await self._get_new_urls(urls)
        
        # Return empty if no new URLs
        if not new_urls:
            return []
            
        # Scrape the content from the URLs
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_urls)
        
        return scraped_content
        
    async def _summarize_content(self, query, content):
        """
        Summarize the extracted content.
        
        Args:
            query: The search query
            content: The extracted content
            
        Returns:
            str: Summarized content
        """
        self.logger.info(f"Summarizing content for query: {query}")
        
        # Skip if no content
        if not content:
            return ""
            
        # Summarize the content using the context manager
        summary = await self.researcher.context_manager.get_similar_content_by_query(
            query, content
        )
        
        return summary
        
    async def _update_search_progress(self, current, total):
        """
        Update the search progress.
        
        Args:
            current: Current number of sub-queries processed
            total: Total number of sub-queries
        """
        if self.researcher.verbose and self.researcher.websocket:
            progress = int((current / total) * 100)
            await stream_output(
                "logs",
                "research_progress",
                f"📊 Research Progress: {progress}%",
                self.researcher.websocket,
                True,
                {
                    "current": current,
                    "total": total,
                    "progress": progress
                }
            )
