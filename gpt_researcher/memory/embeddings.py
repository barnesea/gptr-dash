"""Embedding provider management for GPT Researcher.

This module provides the Memory class that handles embedding generation
across multiple providers (OpenAI, Cohere, Google, Ollama, etc.).

Supported providers:
    - openai: OpenAI embeddings
    - azure_openai: Azure OpenAI embeddings
    - cohere: Cohere embeddings
    - google_vertexai: Google Vertex AI embeddings
    - google_genai: Google Generative AI embeddings
    - fireworks: Fireworks AI embeddings
    - ollama: Local Ollama embeddings
    - together: Together AI embeddings
    - mistralai: Mistral AI embeddings
    - huggingface: HuggingFace embeddings
    - nomic: Nomic embeddings
    - voyageai: Voyage AI embeddings
    - dashscope: DashScope embeddings
    - bedrock: AWS Bedrock embeddings
    - aimlapi: AIML API embeddings
    - custom: Custom OpenAI-compatible API
"""

import os
from typing import Any

from langchain_core.embeddings import Embeddings

OPENAI_EMBEDDING_MODEL = os.environ.get(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

_SUPPORTED_PROVIDERS = {
    "openai",
    "azure_openai",
    "cohere",
    "gigachat",
    "google_vertexai",
    "google_genai",
    "fireworks",
    "ollama",
    "together",
    "mistralai",
    "huggingface",
    "nomic",
    "voyageai",
    "dashscope",
    "custom",
    "bedrock",
    "aimlapi",
    "netmind",
    "openrouter",
    "minimax",
    "nebius",
}


class QueryInstructionEmbeddings(Embeddings):
    """Apply optional, asymmetric query and document prefixes.

    Retrieval-tuned embedding models commonly use an asymmetric format: plain
    Keeping the transformation here makes it work consistently for LangChain
    vector stores and contextual-compression filters.  The legacy instruction
    format remains available for models such as Qwen; retrieval models with
    native ``Query:`` / ``Document:`` conventions can instead use explicit
    prefixes for both sides of the retrieval pair.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        instruction: str = "",
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        self._embeddings = embeddings
        self._query_prefix = query_prefix or (
            f"Instruct: {instruction.strip()}\\nQuery: " if instruction.strip() else ""
        )
        self._document_prefix = document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(
            [f"{self._document_prefix}{text}" for text in texts]
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(f"{self._query_prefix}{text}")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(
            [f"{self._document_prefix}{text}" for text in texts]
        )

    async def aembed_query(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(f"{self._query_prefix}{text}")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embeddings, name)


class Memory:
    """Manages embedding generation for document similarity and retrieval.

    This class provides a unified interface for generating embeddings
    using various providers. It lazily loads provider-specific dependencies
    only when needed.

    Attributes:
        _embeddings: The underlying LangChain embeddings instance.

    Example:
        ```python
        memory = Memory("openai", "text-embedding-3-small")
        embeddings = memory.get_embeddings()
        ```
    """

    def __init__(self, embedding_provider: str, model: str, **embedding_kwargs: Any):
        """Initialize the Memory with a specific embedding provider.

        Args:
            embedding_provider: The name of the embedding provider to use.
                Must be one of the supported providers (openai, cohere, etc.).
            model: The model name/ID to use for embeddings.
            **embedding_kwargs: Additional keyword arguments passed to the
                embedding provider's constructor.

        Raises:
            Exception: If the embedding provider is not supported.
        """
        _embeddings = None
        match embedding_provider:
            case "custom":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("OPENAI_API_KEY", "custom"),
                    openai_api_base=os.getenv(
                        "EMBEDDING_OPENAI_BASE_URL",
                        os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
                    ),  # default for lmstudio
                    check_embedding_ctx_length=False,
                    **embedding_kwargs,
                )  # quick fix for lmstudio
            case "openai":
                from langchain_openai import OpenAIEmbeddings

                # Embedder and chat endpoints can be routed independently.
                # This is needed when a pooling vLLM server is used for
                # embeddings alongside a chat-serving llama-swap endpoint.
                embedding_base_url = os.getenv("EMBEDDING_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
                if "openai_api_base" not in embedding_kwargs and embedding_base_url:
                    embedding_kwargs["openai_api_base"] = embedding_base_url

                _embeddings = OpenAIEmbeddings(model=model, **embedding_kwargs)
            case "azure_openai":
                from langchain_openai import AzureOpenAIEmbeddings

                _embeddings = AzureOpenAIEmbeddings(
                    model=model,
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    openai_api_version=os.environ.get(
                        "AZURE_OPENAI_API_VERSION",
                        os.environ.get("OPENAI_API_VERSION"),
                    ),
                    **embedding_kwargs,
                )
            case "cohere":
                from langchain_cohere import CohereEmbeddings

                _embeddings = CohereEmbeddings(model=model, **embedding_kwargs)
            case "google_vertexai":
                from langchain_google_vertexai import VertexAIEmbeddings

                _embeddings = VertexAIEmbeddings(model=model, **embedding_kwargs)
            case "google_genai":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                _embeddings = GoogleGenerativeAIEmbeddings(
                    model=model, **embedding_kwargs
                )
            case "fireworks":
                from langchain_fireworks import FireworksEmbeddings

                _embeddings = FireworksEmbeddings(model=model, **embedding_kwargs)
            case "gigachat":
                from langchain_gigachat import GigaChatEmbeddings

                _embeddings = GigaChatEmbeddings(model=model, **embedding_kwargs)
            case "ollama":
                from langchain_ollama import OllamaEmbeddings

                _embeddings = OllamaEmbeddings(
                    model=model,
                    base_url=os.environ["OLLAMA_BASE_URL"],
                    **embedding_kwargs,
                )
            case "together":
                from langchain_together import TogetherEmbeddings

                _embeddings = TogetherEmbeddings(model=model, **embedding_kwargs)
            case "netmind":
                from langchain_netmind import NetmindEmbeddings

                _embeddings = NetmindEmbeddings(model=model, **embedding_kwargs)
            case "mistralai":
                from langchain_mistralai import MistralAIEmbeddings

                _embeddings = MistralAIEmbeddings(model=model, **embedding_kwargs)
            case "huggingface":
                from langchain_huggingface import HuggingFaceEmbeddings

                _embeddings = HuggingFaceEmbeddings(model_name=model, **embedding_kwargs)
            case "nomic":
                from langchain_nomic import NomicEmbeddings

                _embeddings = NomicEmbeddings(model=model, **embedding_kwargs)
            case "voyageai":
                from langchain_voyageai import VoyageAIEmbeddings

                _embeddings = VoyageAIEmbeddings(
                    voyage_api_key=os.environ["VOYAGE_API_KEY"],
                    model=model,
                    **embedding_kwargs,
                )
            case "dashscope":
                from langchain_community.embeddings import DashScopeEmbeddings

                _embeddings = DashScopeEmbeddings(model=model, **embedding_kwargs)
            case "bedrock":
                from langchain_aws.embeddings import BedrockEmbeddings

                _embeddings = BedrockEmbeddings(model_id=model, **embedding_kwargs)
            case "aimlapi":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("AIMLAPI_API_KEY"),
                    openai_api_base=os.getenv("AIMLAPI_BASE_URL", "https://api.aimlapi.com/v1"),
                    **embedding_kwargs,
                )
            case "openrouter":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                    openai_api_base="https://openrouter.ai/api/v1",
                    **embedding_kwargs,
                )
            case "minimax":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("MINIMAX_API_KEY"),
                    openai_api_base="https://api.minimax.io/v1",
                    **embedding_kwargs,
                )
            case "nebius":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("NEBIUS_API_KEY"),
                    openai_api_base=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
                    **embedding_kwargs,
                )
            case _:
                raise Exception("Embedding not found.")

        instruction = os.getenv("EMBEDDING_QUERY_INSTRUCTION", "").strip()
        query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", "")
        document_prefix = os.getenv("EMBEDDING_DOCUMENT_PREFIX", "")
        self._embeddings = (
            QueryInstructionEmbeddings(
                _embeddings,
                instruction=instruction,
                query_prefix=query_prefix,
                document_prefix=document_prefix,
            )
            if instruction or query_prefix or document_prefix
            else _embeddings
        )

    def get_embeddings(self):
        """Get the configured embeddings instance.

        Returns:
            The LangChain embeddings instance configured for this Memory.
        """
        return self._embeddings
