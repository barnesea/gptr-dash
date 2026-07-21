# GPT Researcher MCP Server

This directory contains the MCP server that ships with `gptr-dash`. It exposes
GPT Researcher as MCP tools and resources while importing the local
`gpt_researcher` package from the repository root.

## Docker Compose

The root Compose stack starts the MCP server over SSE by default:

```bash
docker compose up --build gptr-mcp
```

Default endpoints:

- Health: http://localhost:8001/health
- SSE: http://localhost:8001/sse
- Messages: http://localhost:8001/messages/?session_id=YOUR_SESSION_ID
- Streamable HTTP: http://localhost:8001/mcp when `MCP_TRANSPORT=streamable-http`

## Environment

Set these in the root `.env` file:

```bash
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
MCP_TRANSPORT=sse
MCP_PORT=8001
MCP_MAX_CONCURRENT_DEEP_RESEARCH=1
MCP_ENABLE_RECURSIVE_DEEP_RESEARCH=true
# Optional for Streamable HTTP clients such as Open WebUI:
# MCP_TRANSPORT=streamable-http
# MCP_PATH=/mcp
```

Set `MCP_MAX_CONCURRENT_DEEP_RESEARCH=0` to disable the limit. The limit applies
only to the expensive `deep_research` tool; lighter MCP tools and health checks
continue to run while deep research calls wait for a slot.

Set `MCP_ENABLE_RECURSIVE_DEEP_RESEARCH=false` to make the MCP `deep_research`
tool use the lighter standard research workflow instead of GPT Researcher's
recursive deep research mode.

Optional GPT Researcher settings such as `OPENAI_BASE_URL`,
`LANGCHAIN_API_KEY`, retriever settings, and model configuration are inherited
from the same environment.

## Tools

- `deep_research`: conduct deep web research and return context plus sources.
- `quick_search`: perform a faster web search with snippets.
- `write_report`: generate a report from a previous research session.
- `get_research_sources`: return the sources for a research session.
- `get_research_context`: return the full context for a research session.

## Local STDIO Mode

For clients that spawn MCP servers directly, run from the repo root:

```bash
MCP_TRANSPORT=stdio python mcp-server/server.py
```

Example local client command:

```json
{
  "mcpServers": {
    "gptr-dash": {
      "command": "python",
      "args": ["/home/pichiad/workspaces/gptr-dash/mcp-server/server.py"],
      "env": {
        "OPENAI_API_KEY": "your-openai-key",
        "TAVILY_API_KEY": "your-tavily-key",
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```
