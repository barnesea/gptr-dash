"""Context compression utilities for GPT Researcher.

This module provides classes for compressing and retrieving relevant
context from documents using embeddings and similarity filtering.

The compression pipeline:
1. Splits documents into chunks
2. Filters chunks by embedding similarity to the query
3. Returns the most relevant chunks as context

Classes:
    VectorstoreCompressor: Retrieves context from a vector store.
    ContextCompressor: Compresses raw documents using embedding similarity.
    WrittenContentCompressor: Compresses previously written content sections.
"""

import asyncio
from dataclasses import asdict, dataclass
import math
import os
from typing import Any, Optional

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..memory.embeddings import OPENAI_EMBEDDING_MODEL
from ..actions.source_selection import (
    has_meaningful_query_anchor,
    has_requested_evidence_relation,
    requires_supported_evidence_relation,
    source_quality_tier,
)
from ..prompts import PromptFamily
from ..utils.costs import estimate_embedding_cost
from ..vector_store import VectorStoreWrapper
from .retriever import SearchAPIRetriever, SectionRetriever


@dataclass
class CompressionResult:
    """Compressed evidence plus retrieval diagnostics for trajectory analysis."""

    context: str
    chunk_count: int
    top_score: float
    accepted_count: int
    rescue_used: bool
    threshold: float
    rescue_floor: float | None
    accepted_by_source: dict[str, int]
    rescue_mode: str = ""

    def diagnostics(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("context", None)
        return payload


class VectorstoreCompressor:
    """Retrieves and compresses context from a vector store.

    Uses similarity search on an existing vector store to find
    relevant documents for a given query.

    Attributes:
        vector_store: The vector store wrapper to search.
        max_results: Maximum number of results to return.
        filter: Optional filter for vector store queries.
    """

    def __init__(
        self,
        vector_store: VectorStoreWrapper,
        max_results: int = 7,
        filter: Optional[dict] = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """Initialize the VectorstoreCompressor.

        Args:
            vector_store: The vector store to search.
            max_results: Maximum number of results to return.
            filter: Optional filter dictionary for queries.
            prompt_family: Prompt family for formatting output.
            **kwargs: Additional keyword arguments.
        """
        self.vector_store = vector_store
        self.max_results = max_results
        self.filter = filter
        self.kwargs = kwargs
        self.prompt_family = prompt_family

    async def async_get_context(self, query: str, max_results: int = 5) -> str:
        """Get relevant context from the vector store.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            Formatted string of relevant document content.
        """
        results = await self.vector_store.asimilarity_search(query=query, k=max_results, filter=self.filter)
        return self.prompt_family.pretty_print_docs(results)


class ContextCompressor:
    """Compresses raw documents to extract relevant context.

    Uses embedding similarity to filter document chunks and return
    only the most relevant content for a given query.

    Attributes:
        documents: List of documents to compress.
        embeddings: Embedding model for similarity calculation.
        max_results: Maximum number of results to return.
        similarity_threshold: Minimum similarity score for inclusion.
    """

    def __init__(
        self,
        documents,
        embeddings,
        max_results: int = 5,
        similarity_threshold: float | None = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """Initialize the ContextCompressor.

        Args:
            documents: List of documents to compress.
            embeddings: Embedding model instance.
            max_results: Maximum number of results to return.
            similarity_threshold: Minimum similarity score for inclusion.
                Falls back to the SIMILARITY_THRESHOLD env var when not given.
            prompt_family: Prompt family for formatting output.
            **kwargs: Additional keyword arguments.
        """
        self.max_results = max_results
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        if similarity_threshold is None:
            similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.35))
        self.similarity_threshold = similarity_threshold
        self.prompt_family = prompt_family

    def __get_contextual_retriever(self):
        """Build the contextual compression retriever pipeline.

        Returns:
            A ContextualCompressionRetriever configured with text splitting
            and embedding-based filtering.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(embeddings=self.embeddings,
                                            similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SearchAPIRetriever(
            pages=self.documents
        )
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> str:
        """Get relevant context from documents asynchronously.

        Optimization: Skip expensive compression pipeline for small document sets.
        When documents are already concise, directly use them without embedding-based filtering.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            cost_callback: Optional callback for tracking embedding costs.

        Returns:
            Formatted string of relevant document content.
        """
        # Optimization: Calculate total content size
        total_chars = sum(len(str(doc.get('raw_content', ''))) for doc in self.documents)
        chunk_threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))

        # If total content is small, skip expensive compression and return directly
        if total_chars < chunk_threshold and len(self.documents) <= max_results:
            # Fast path: no compression needed
            # Map scraper/retriever dict keys into metadata that pretty_print_docs expects.
            # Raw dicts use `url`; SearchAPIRetriever / pretty_print use `source`.
            direct_docs = [
                Document(
                    page_content=doc.get('raw_content', '') or '',
                    metadata={
                        "title": doc.get("title", "") or "",
                        "source": doc.get("source") or doc.get("url") or "",
                    },
                )
                for doc in self.documents[:max_results]
            ]
            return self.prompt_family.pretty_print_docs(direct_docs, max_results)

        # Standard path: use compression for large content
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query, **self.kwargs)
        return self.prompt_family.pretty_print_docs(relevant_docs, max_results)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _source_documents(self) -> list[Document]:
        documents: list[Document] = []
        for item in self.documents:
            url = str(item.get("source") or item.get("url") or item.get("href") or "")
            content = str(item.get("raw_content") or item.get("content") or "")
            if not content.strip():
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "title": str(item.get("title") or ""),
                        "source": url,
                        "source_tier": source_quality_tier(
                            {
                                "url": url,
                                "title": item.get("title", ""),
                                "_gptr_source_tier": item.get(
                                    "_gptr_source_tier", ""
                                ),
                            }
                        ),
                    },
                )
            )
        return documents

    async def async_get_context_with_diagnostics(
        self,
        query: str,
        *,
        max_results: int = 10,
        cost_callback=None,
        adaptive: bool = False,
        rescue_floor: float = 0.30,
        max_chunks_per_source: int = 3,
        rescue_chunks_per_source: int = 2,
    ) -> CompressionResult:
        """Return scored chunks and selectively rescue trusted pages.

        Ordinary callers retain the legacy compressor. Deep branches opt into
        explicit scores so a high-quality page is not discarded wholesale when
        its best chunk narrowly misses the normal threshold.
        """
        if not adaptive:
            context = await self.async_get_context(
                query=query,
                max_results=max_results,
                cost_callback=cost_callback,
            )
            return CompressionResult(
                context=context,
                chunk_count=0,
                top_score=0.0,
                accepted_count=1 if context else 0,
                rescue_used=False,
                threshold=self.similarity_threshold,
                rescue_floor=None,
                accepted_by_source={},
                rescue_mode="",
            )

        source_documents = self._source_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = [
            chunk
            for chunk in splitter.split_documents(source_documents)
            if chunk.page_content.strip()
        ]
        if not chunks:
            return CompressionResult(
                context="",
                chunk_count=0,
                top_score=0.0,
                accepted_count=0,
                rescue_used=False,
                threshold=self.similarity_threshold,
                rescue_floor=rescue_floor,
                accepted_by_source={},
                rescue_mode="",
            )

        if cost_callback:
            cost_callback(
                estimate_embedding_cost(
                    model=OPENAI_EMBEDDING_MODEL,
                    docs=[{"raw_content": chunk.page_content} for chunk in chunks],
                )
            )
        query_vector, document_vectors = await asyncio.gather(
            self.embeddings.aembed_query(query),
            self.embeddings.aembed_documents(
                [chunk.page_content for chunk in chunks]
            ),
        )
        scored = sorted(
            [
                (self._cosine(query_vector, vector), index, chunk)
                for index, (chunk, vector) in enumerate(zip(chunks, document_vectors))
            ],
            key=lambda item: (-item[0], item[1]),
        )
        top_score = scored[0][0] if scored else 0.0
        normal = [item for item in scored if item[0] >= self.similarity_threshold]
        rescue_used = False
        rescue_mode = ""
        candidates = normal
        per_source_cap = max(1, max_chunks_per_source)
        if not normal:
            rescue_used = True
            candidates = [
                item
                for item in scored
                if (
                    item[0] >= rescue_floor
                    and item[2].metadata.get("source_tier")
                    in {"primary", "reputable"}
                )
            ]
            per_source_cap = max(1, rescue_chunks_per_source)
            if candidates:
                rescue_mode = "similarity_floor"
            else:
                # Embedding distributions can shift by domain and chunk style.
                # When a verified primary/reputable chunk visibly contains
                # multiple query anchors, recover it locally rather than
                # spending another search or returning an empty branch.
                candidates = [
                    item
                    for item in scored
                    if (
                        item[2].metadata.get("source_tier")
                        in {"primary", "reputable"}
                        and has_meaningful_query_anchor(
                            query,
                            {
                                "url": item[2].metadata.get("source", ""),
                                "title": item[2].metadata.get("title", ""),
                                "body": item[2].page_content,
                            },
                        )
                    )
                ]
                if candidates:
                    rescue_mode = "verified_lexical_anchor"

        # Semantic similarity can favor broad background passages over the
        # actual relationship requested by the user. For supported relation
        # types, promote verified chunks that contain a concrete claim even
        # when ordinary threshold matches already exist.
        if requires_supported_evidence_relation(query):
            relation_candidates = [
                item
                for item in scored
                if (
                    item[2].metadata.get("source_tier")
                    in {"primary", "reputable"}
                    and has_requested_evidence_relation(
                        query,
                        item[2].page_content,
                    )
                )
            ]
            if relation_candidates:
                relation_keys = {
                    (
                        item[1],
                        str(item[2].metadata.get("source") or ""),
                    )
                    for item in relation_candidates
                }
                candidates = relation_candidates + [
                    item
                    for item in candidates
                    if (
                        item[1],
                        str(item[2].metadata.get("source") or ""),
                    )
                    not in relation_keys
                ]
                rescue_used = True
                rescue_mode = "verified_relation_anchor"

        accepted: list[Document] = []
        accepted_by_source: dict[str, int] = {}
        for _score, _index, chunk in candidates:
            source = str(chunk.metadata.get("source") or "")
            if accepted_by_source.get(source, 0) >= per_source_cap:
                continue
            accepted.append(chunk)
            accepted_by_source[source] = accepted_by_source.get(source, 0) + 1
            if len(accepted) >= max(1, max_results):
                break

        return CompressionResult(
            context=self.prompt_family.pretty_print_docs(accepted, max_results),
            chunk_count=len(chunks),
            top_score=round(top_score, 6),
            accepted_count=len(accepted),
            rescue_used=rescue_used and bool(accepted),
            threshold=self.similarity_threshold,
            rescue_floor=rescue_floor,
            accepted_by_source=accepted_by_source,
            rescue_mode=rescue_mode if accepted else "",
        )


class WrittenContentCompressor:
    """Compresses previously written content sections.

    Specialized compressor for finding relevant sections from
    previously written report content, preserving section titles
    and structure.

    Attributes:
        documents: List of written content sections.
        embeddings: Embedding model for similarity calculation.
        similarity_threshold: Minimum similarity score for inclusion.
    """

    def __init__(self, documents, embeddings, similarity_threshold: float, **kwargs):
        """Initialize the WrittenContentCompressor.

        Args:
            documents: List of written content sections.
            embeddings: Embedding model instance.
            similarity_threshold: Minimum similarity score for inclusion.
            **kwargs: Additional keyword arguments.
        """
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    def __get_contextual_retriever(self):
        """Build the contextual compression retriever for sections.

        Returns:
            A ContextualCompressionRetriever configured for section retrieval.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(embeddings=self.embeddings,
                                            similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SectionRetriever(
            sections=self.documents
        )
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    def __pretty_docs_list(self, docs, top_n: int) -> list[str]:
        """Format documents as a list of title/content strings.

        Args:
            docs: List of documents to format.
            top_n: Maximum number of documents to include.

        Returns:
            List of formatted document strings.
        """
        return [f"Title: {d.metadata.get('section_title')}\nContent: {d.page_content}\n" for i, d in enumerate(docs) if i < top_n]

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> list[str]:
        """Get relevant written content sections asynchronously.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            cost_callback: Optional callback for tracking embedding costs.

        Returns:
            List of formatted section strings.
        """
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query, **self.kwargs)
        return self.__pretty_docs_list(relevant_docs, max_results)
