from .base import BaseConfig

DEFAULT_CONFIG: BaseConfig = {
    "RETRIEVER": "tavily",
    "EMBEDDING": "openai:text-embedding-3-small",
    "SIMILARITY_THRESHOLD": 0.42,
    "FAST_LLM": "openai:gpt-5.4-mini",
    "SMART_LLM": "openai:gpt-5.4",  # Has support for long responses (2k+ words).
    "STRATEGIC_LLM": "openai:gpt-5.4",  # Reasoning model used for planning; tune REASONING_EFFORT for speed vs. depth.
    # Output token limits. For reasoning models (the default gpt-5.x family)
    # these map to max_completion_tokens, which also covers reasoning tokens -
    # hence the generous headroom on top of the visible output.
    "FAST_TOKEN_LIMIT": 6000,
    "SMART_TOKEN_LIMIT": 12000,
    "STRATEGIC_TOKEN_LIMIT": 8000,
    "BROWSE_CHUNK_MAX_LENGTH": 8192,
    "CURATE_SOURCES": False,
    "SUMMARY_TOKEN_LIMIT": 700,
    "TEMPERATURE": 0.4,
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "MAX_SEARCH_RESULTS_PER_QUERY": 5,
    # Evidence-grounded planning and pre-scrape selection are opt-in at the
    # package level so existing deployments retain their current behavior.
    "PLANNING_EVIDENCE_ENABLED": False,
    "PLANNING_SEARCH_RESULTS": 8,
    "PRE_SCRAPE_SOURCE_CURATION": False,
    "PRE_SCRAPE_MAX_SOURCES_PER_QUERY": 3,
    # Verify that fetched content still matches its selected SERP card before
    # it reaches the evidence context.  Opt-in preserves package behavior.
    "POST_SCRAPE_SOURCE_INTEGRITY": False,
    "SOURCE_CURATION_POLICY": "balanced",
    # v2 preserves preliminary evidence across the job, renders compact
    # entity-first queries, and judges every selected source before synthesis.
    # "legacy" is an immediate rollback path.
    "RETRIEVAL_PIPELINE_MODE": "legacy",
    "SOURCE_EVIDENCE_JUDGE_MODE": "all",
    "SOURCE_EVIDENCE_JUDGE_FALLBACK": "hybrid",
    # Retain relevant but non-conclusive evidence with explicit labels instead
    # of silently treating it as synthesis-ready or discarding it.
    "LABELED_EVIDENCE_POOL_ENABLED": False,
    "LABELED_EVIDENCE_MIN_CONFIDENCE": 0.20,
    "CANONICAL_CONTENT_RESOLUTION": True,
    # Define the named subject and its supporting concepts with a small set of
    # preliminary searches before planning deeper work. Deployments can enable
    # this independently from retrieval pipeline v2 for a clean rollback.
    "SUBJECT_GROUNDING_ENABLED": False,
    "SUBJECT_GROUNDING_MAX_SUBJECTS": 4,
    "SUBJECT_GROUNDING_RESULTS_PER_QUERY": 3,
    "SUBJECT_GROUNDING_SEARCH_CONCURRENCY": 4,
    "MEMORY_BACKEND": "local",
    "TOTAL_WORDS": 1200,
    "REPORT_FORMAT": "APA",
    "MAX_ITERATIONS": 3,
    "AGENT_ROLE": None,
    "SCRAPER": "bs",
    "MAX_SCRAPER_WORKERS": 15,
    "SCRAPER_RATE_LIMIT_DELAY": 0.0,  # Minimum seconds between scraper requests (0 = no limit, useful for API rate limiting)
    "MAX_SUBTOPICS": 3,
    "LANGUAGE": "english",
    "REPORT_SOURCE": "web",
    "DOC_PATH": "./my-docs",
    "PROMPT_FAMILY": "default",
    "LLM_KWARGS": {},
    "EMBEDDING_KWARGS": {},
    "VERBOSE": False,
    # Deep research specific settings
    "DEEP_RESEARCH_BREADTH": 3,
    "DEEP_RESEARCH_DEPTH": 2,
    "DEEP_RESEARCH_CONCURRENCY": 4,
    # Recursive branches are deliberately focused: the deep-research tree owns
    # breadth, so branch researchers must not multiply it again.
    "DEEP_RESEARCH_FOCUSED_RETRIEVAL": True,
    # "auto" preserves DEEP_RESEARCH_FOCUSED_RETRIEVAL as the compatibility
    # switch. Explicit values decouple tree expansion from branch retrieval for
    # controlled trajectory comparisons.
    "DEEP_RESEARCH_TREE_POLICY": "auto",
    "DEEP_RESEARCH_BRANCH_MODE": "auto",
    "DEEP_RESEARCH_MAX_DEEPENED_BRANCHES": 1,
    "DEEP_RESEARCH_MIN_DEEPENING_SCORE": 8,
    # "strongest" preserves the earlier ranked-tree behavior. Deployments can
    # use gap_first to spend follow-up work on the most important unresolved
    # aspect before adding detail to already well-supported branches.
    "DEEP_RESEARCH_DEEPENING_STRATEGY": "strongest",
    "DEEP_RESEARCH_SOURCE_STANDARDS": True,
    "DEEP_RESEARCH_DIRECT_URL_SEED": False,
    # auto uses the fast selector only when deterministic evidence is genuinely
    # ambiguous; llm and deterministic are immediate rollback modes.
    "SOURCE_SELECTOR_MODE": "auto",
    # Job-scoped JSONL trajectories are opt-in for package users. Deployments
    # can persist them by mounting RESEARCH_TRAJECTORY_DIR.
    "RESEARCH_TRAJECTORY_ENABLED": False,
    "RESEARCH_TRAJECTORY_DIR": "data/trajectories",
    # off and shadow execute the current internal 2/3 ranked/focused policy.
    # shadow additionally calculates and reports the recommended duration policy.
    "RESEARCH_DURATION_CONTROLLER_MODE": "off",
    "RESEARCH_DURATION_DEFAULT_SECONDS": 60,
    "RESEARCH_DURATION_MIN_SECONDS": 15,
    "RESEARCH_DURATION_MAX_SECONDS": 600,
    "RESEARCH_BUDGET_CALIBRATION_ENABLED": True,
    "RESEARCH_BUDGET_CALIBRATION_MIN_SAMPLES": 10,
    # Remote PDFs are a common source of long-tail latency. These limits apply
    # to deep-research branches; ordinary browsing keeps the scraper defaults.
    "DEEP_RESEARCH_PDF_CONNECT_TIMEOUT_SECONDS": 3.0,
    "DEEP_RESEARCH_PDF_TOTAL_TIMEOUT_SECONDS": 8.0,
    "DEEP_RESEARCH_PDF_MAX_BYTES": 32 * 1024 * 1024,
    "DEEP_RESEARCH_ADAPTIVE_COMPRESSION": False,
    "DEEP_RESEARCH_SIMILARITY_RESCUE_FLOOR": 0.30,
    "DEEP_RESEARCH_MAX_CONTEXT_CHUNKS": 10,
    "DEEP_RESEARCH_MAX_CHUNKS_PER_SOURCE": 3,
    "DEEP_RESEARCH_FALLBACK_CORROBORATION_ENABLED": False,
    "DEEP_RESEARCH_FALLBACK_CORROBORATION": 2,
    
    # MCP retriever specific settings
    "MCP_SERVERS": [],  # List of predefined MCP server configurations
    "MCP_AUTO_TOOL_SELECTION": True,  # Whether to automatically select the best tool for a query
    "MCP_ALLOWED_ROOT_PATHS": [],  # List of allowed root paths for local file access
    "MCP_STRATEGY": "fast",  # MCP execution strategy: "fast", "deep", "disabled"
    "REASONING_EFFORT": "medium",
    
    # Image generation settings (optional - requires GOOGLE_API_KEY)
    # Free tier models: gemini-2.5-flash-image, gemini-2.0-flash-exp-image-generation
    # Paid tier models: imagen-4.0-generate-001, imagen-4.0-fast-generate-001
    "IMAGE_GENERATION_MODEL": "models/gemini-2.5-flash-image",
    "IMAGE_GENERATION_MAX_IMAGES": 3,  # Maximum number of images to generate per report
    "IMAGE_GENERATION_ENABLED": False,  # Master switch for inline image generation
    "IMAGE_GENERATION_STYLE": "dark",  # Image style: "dark" (matches app theme), "light", or "auto"
    "IMAGE_GENERATION_PROVIDER": "google",  # Image provider: "google" or "modelslab"

    # llama-swap defaults for local OpenAI-compatible model routing.
    "LLAMA_SWAP_URL": "http://localhost:8080",
    "LLAMA_SWAP_ENABLED": True,
    "LLAMA_SWAP_TIMEOUT": 1.0,
}
