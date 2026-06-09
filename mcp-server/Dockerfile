FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching.
COPY requirements.txt ./requirements.txt
COPY mcp-server/requirements.txt ./mcp-server/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --prefer-binary && \
    pip install --no-cache-dir -r mcp-server/requirements.txt --prefer-binary

# Copy application code
COPY . .

# Set environment variables for Docker
ENV MCP_TRANSPORT=sse
ENV MCP_PORT=8001
ENV DOCKER_CONTAINER=true
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=7s --retries=3 \
    CMD curl -f http://localhost:${MCP_PORT}/health || exit 1

# Run the server
CMD ["python", "mcp-server/server.py"] 
