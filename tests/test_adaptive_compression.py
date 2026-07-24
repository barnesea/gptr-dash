import math

import pytest

from gpt_researcher.context.compression import ContextCompressor


class FakeEmbeddings:
    async def aembed_query(self, _text):
        return [1.0, 0.0]

    async def aembed_documents(self, texts):
        vectors = []
        for text in texts:
            score = 0.35 if "recoverable" in text else 0.10
            vectors.append([score, math.sqrt(1 - score * score)])
        return vectors


class FakePrompt:
    @staticmethod
    def pretty_print_docs(documents, max_results=10):
        return "\n".join(
            f"Source: {document.metadata.get('source')}\n{document.page_content}"
            for document in documents[:max_results]
        )


class BackgroundBiasedEmbeddings:
    async def aembed_query(self, _text):
        return [1.0, 0.0]

    async def aembed_documents(self, texts):
        vectors = []
        for text in texts:
            score = 0.50 if "shell morphology" in text else 0.20
            vectors.append([score, math.sqrt(1 - score * score)])
        return vectors


@pytest.mark.asyncio
async def test_trusted_page_recovers_top_chunks_below_normal_threshold():
    documents = [
        {
            "url": "https://docs.example.com/product",
            "title": "Official product documentation",
            "raw_content": ("recoverable evidence " * 900),
        }
    ]
    compressor = ContextCompressor(
        documents,
        FakeEmbeddings(),
        similarity_threshold=0.42,
        prompt_family=FakePrompt,
    )

    result = await compressor.async_get_context_with_diagnostics(
        "product evidence",
        adaptive=True,
        rescue_floor=0.30,
        max_results=10,
        max_chunks_per_source=3,
    )

    assert result.top_score == pytest.approx(0.35)
    assert result.rescue_used is True
    assert result.accepted_count == 2
    assert result.chunk_count > result.accepted_count


@pytest.mark.asyncio
async def test_unverified_fallback_page_is_not_rescued():
    compressor = ContextCompressor(
        [
            {
                "url": "https://unknown.example/article",
                "title": "Article",
                "raw_content": "recoverable evidence " * 900,
            }
        ],
        FakeEmbeddings(),
        similarity_threshold=0.42,
        prompt_family=FakePrompt,
    )

    result = await compressor.async_get_context_with_diagnostics(
        "product evidence",
        adaptive=True,
        rescue_floor=0.30,
    )

    assert result.rescue_used is False
    assert result.accepted_count == 0
    assert result.context == ""


@pytest.mark.asyncio
async def test_verified_page_uses_lexical_recovery_below_similarity_floor():
    compressor = ContextCompressor(
        [
            {
                "url": "https://www.nps.gov/example/turtles",
                "title": "Turtle nest predators",
                "raw_content": (
                    "Natural predators of turtle nests and hatchlings include "
                    "raccoons and foxes. "
                )
                * 80,
            }
        ],
        FakeEmbeddings(),
        similarity_threshold=0.42,
        prompt_family=FakePrompt,
    )

    result = await compressor.async_get_context_with_diagnostics(
        "natural predators of turtle nests and hatchlings",
        adaptive=True,
        rescue_floor=0.30,
    )

    assert result.top_score == pytest.approx(0.10)
    assert result.rescue_used is True
    assert result.rescue_mode == "verified_relation_anchor"
    assert result.accepted_count > 0


@pytest.mark.asyncio
async def test_relation_chunk_is_promoted_when_background_scores_higher():
    compressor = ContextCompressor(
        [
            {
                "url": "https://www.nps.gov/example/turtles",
                "title": "Turtle biology and predators",
                "raw_content": (
                    ("Sea turtle shell morphology and swimming adaptations. " * 35)
                    + ("Predators of sea turtles include sharks and orcas. " * 35)
                ),
            }
        ],
        BackgroundBiasedEmbeddings(),
        similarity_threshold=0.42,
        prompt_family=FakePrompt,
    )

    result = await compressor.async_get_context_with_diagnostics(
        "natural predators of sea turtles",
        adaptive=True,
        rescue_floor=0.30,
        max_chunks_per_source=3,
    )

    assert result.rescue_used is True
    assert result.rescue_mode == "verified_relation_anchor"
    assert "sharks and orcas" in result.context
