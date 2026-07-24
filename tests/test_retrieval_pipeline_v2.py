from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpt_researcher.actions.retrieval_pipeline import (
    EvidenceCandidateLedger,
    canonicalize_url,
    evidence_role,
    lexical_collision,
    missing_evidence_roles,
    normalize_aspect_contract,
    normalized_tokens,
    render_compact_query,
    roles_cover_aspect,
)
from gpt_researcher.actions.source_selection import (
    canonical_source_alternatives,
    has_requested_evidence_relation,
    rebalance_evidence_roles,
    source_site,
)
from gpt_researcher.skills.researcher import ResearchConductor


def krea_aspect():
    return normalize_aspect_contract(
        {
            "id": "workflow",
            "intent": "procedural",
            "question": (
                "How can style LoRAs or LoKRs be trained for Krea 2 with "
                "Ostris AI Toolkit?"
            ),
            "relation": "style LoRA LoKR training workflow",
            "entities": [
                {"name": "Krea 2", "aliases": ["Krea2"], "exact": True},
                {
                    "name": "Ostris AI Toolkit",
                    "aliases": ["ai-toolkit"],
                    "exact": True,
                },
            ],
            "evidence_roles": ["first_party", "practitioner"],
        },
        "How do I make style LoRAs or LoKRs for Krea 2 using Ostris AI Toolkit?",
    )


def test_compact_query_is_entity_first_and_bounded():
    query = render_compact_query(
        krea_aspect(),
        evidence_role="first_party",
    )

    assert '"Krea 2"' in query
    assert '"Ostris AI Toolkit"' in query
    assert "LoKR" in query or "lokr" in query
    assert "How can" not in query
    assert len(normalized_tokens(query)) <= 12


def test_topic_anchors_are_not_promoted_to_exact_entities():
    aspect = normalize_aspect_contract(
        {
            "question": "Explain a broad biological process",
            "original_query_anchors": ["broad", "biological", "process"],
            "entities": [],
        },
        "Explain a broad biological process",
    )

    assert aspect["entities"] == []


def test_entity_guard_detects_generic_verb_collision():
    candidates = [
        {
            "url": "https://amtrak.com/train-routes",
            "title": "Train routes",
            "body": "Train schedule and route guide",
        },
        {
            "url": "https://rail.example/training",
            "title": "Train travel",
            "body": "Training and train station information",
        },
    ]

    assert lexical_collision(krea_aspect(), candidates)


def test_neutral_host_provenance_is_relative_to_the_named_subject():
    aspect = krea_aspect()

    assert evidence_role(
        {"url": "https://github.com/ostris/ai-toolkit"},
        aspect,
    ) == "first_party"
    repo_aspect = normalize_aspect_contract(
        {
            "entities": [
                {"name": "Mage-Flow", "exact": True}
            ],
            "relation": "architecture",
        },
        "Research Mage-Flow",
    )
    assert evidence_role(
        {"url": "https://github.com/microsoft/Mage-Flow"},
        repo_aspect,
    ) == "first_party"
    assert evidence_role(
        {
            "url": (
                "https://huggingface.co/ostris/"
                "krea2_turbo_style_reference"
            )
        },
        aspect,
    ) == "first_party"
    assert evidence_role(
        {"url": "https://github.com/CaptainGrock/Krea2Trainer"},
        aspect,
    ) == "practitioner"
    assert evidence_role(
        {
            "url": (
                "https://huggingface.co/ostris/"
                "krea2_turbo_style_reference/discussions/4"
            )
        },
        aspect,
    ) == "practitioner"
    assert evidence_role(
        {"url": "https://github.com/ostris/ai-toolkit/issues/123"},
        aspect,
    ) == "practitioner"


def test_role_coverage_drives_the_next_search_portfolio():
    aspect = krea_aspect()
    official = {
        "url": "https://github.com/ostris/ai-toolkit",
        "title": "Ostris AI Toolkit",
    }
    practitioner = {
        "url": "https://github.com/CaptainGrock/Krea2Trainer",
        "title": "Krea2 trainer workflow",
    }

    assert missing_evidence_roles(aspect, [official]) == ["practitioner"]
    assert roles_cover_aspect(aspect, [official, practitioner])


def test_source_cap_preserves_requested_evidence_role_mix():
    candidates = [
        {
            "url": "https://official.example/krea",
            "title": "Krea 2 Ostris AI Toolkit style LoRA training",
            "body": "Official Krea 2 style LoRA training workflow",
            "_gptr_evidence_role": "first_party",
        },
        {
            "url": "https://paper.example/krea",
            "title": "Krea 2 style LoRA training study",
            "body": "Krea 2 Ostris AI Toolkit training evidence",
            "_gptr_evidence_role": "original",
        },
        {
            "url": "https://practitioner.example/krea",
            "title": "Krea 2 style LoRA training workflow",
            "body": "Hands-on Ostris AI Toolkit LoRA workflow",
            "_gptr_evidence_role": "practitioner",
        },
    ]

    balanced = rebalance_evidence_roles(
        "Krea 2 Ostris AI Toolkit style LoRA training",
        candidates,
        candidates[:2],
        ["first_party", "practitioner"],
        2,
    )

    assert {
        item["_gptr_evidence_role"] for item in balanced
    } == {"first_party", "practitioner"}


def test_candidate_ledger_fuses_discoveries_and_excludes_failed_sources():
    ledger = EvidenceCandidateLedger()
    first = {
        "url": (
            "https://huggingface.co/ostris/"
            "krea2_turbo_style_reference?utm_source=test"
        ),
        "title": "Krea 2 style reference",
        "body": "Ostris Krea 2 style LoRA training reference",
    }
    second = {
        **first,
        "url": (
            "https://huggingface.co/ostris/"
            "krea2_turbo_style_reference"
        ),
    }
    ledger.register([first], stage="preliminary", query="Krea 2")
    ledger.register(
        [second],
        stage="aspect_search",
        query="Krea 2 style LoRA",
    )

    snapshot = ledger.snapshot()
    assert snapshot["candidate_count"] == 1
    assigned = ledger.candidates_for_aspect(krea_aspect(), limit=3)
    assert len(assigned) == 1
    assert assigned[0]["_gptr_discovery_count"] == 2
    assert assigned[0]["_gptr_fusion_score"] > 1

    ledger.record_judgment(
        second["url"],
        {
            "aspect_id": "workflow",
            "supports_aspect": False,
            "accepted_for_synthesis": False,
        },
    )
    assert ledger.candidates_for_aspect(krea_aspect(), limit=3) == []
    other_aspect = {
        **krea_aspect(),
        "id": "licensing",
        "relation": "license terms",
    }
    assert len(
        ledger.candidates_for_aspect(other_aspect, limit=3)
    ) == 1
    assert ledger.candidates_for_aspect(
        other_aspect,
        limit=3,
        exclude_urls=[second["url"]],
    ) == []


def test_canonicalization_and_keyless_resolvers_are_bounded():
    assert canonicalize_url(
        "https://Example.com/path/?utm_source=x&keep=1#section"
    ) == "https://example.com/path?keep=1"
    assert canonical_source_alternatives(
        "https://github.com/ostris/ai-toolkit"
    ) == [
        "https://raw.githubusercontent.com/ostris/ai-toolkit/main/README.md",
        "https://raw.githubusercontent.com/ostris/ai-toolkit/master/README.md",
    ]
    assert canonical_source_alternatives(
        "https://huggingface.co/ostris/model"
    ) == [
        "https://huggingface.co/ostris/model/resolve/main/README.md"
    ]
    assert len(
        canonical_source_alternatives("https://doi.org/10.1000/example")
    ) == 2
    assert len(
        canonical_source_alternatives(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC12345/"
        )
    ) == 2
    assert source_site("https://docs.example.com/a") == source_site(
        "https://blog.example.com/b"
    )


@pytest.mark.parametrize(
    ("query", "relation", "entities", "good_text", "bad_text"),
    [
        (
            "natural predators of freshwater turtles",
            "natural predation",
            [{"name": "freshwater turtles", "exact": True}],
            "Freshwater turtles experience natural predation by foxes.",
            "Freshwater turtles inhabit ponds and bask on logs.",
        ),
        (
            "Krea 2 style LoRA training with Ostris AI Toolkit",
            "style LoRA training workflow",
            [
                {"name": "Krea 2", "aliases": ["Krea2"], "exact": True},
                {
                    "name": "Ostris AI Toolkit",
                    "aliases": ["ai-toolkit"],
                    "exact": True,
                },
            ],
            (
                "https://github.com/ostris/ai-toolkit Krea2 style LoRA "
                "training workflow"
            ),
            "Train schedules and station workflow documentation.",
        ),
        (
            "Medicinex adverse effects",
            "adverse effects",
            [{"name": "Medicinex", "exact": True}],
            "Medicinex adverse effects include dizziness and nausea.",
            "Medicinex is supplied as a 10 mg tablet.",
        ),
    ],
)
def test_hybrid_relation_guard_generalizes_across_domains(
    query, relation, entities, good_text, bad_text
):
    assert has_requested_evidence_relation(
        query,
        good_text,
        relation=relation,
        entity_anchors=entities,
    )
    assert not has_requested_evidence_relation(
        query,
        bad_text,
        relation=relation,
        entity_anchors=entities,
    )


def test_evidence_judgment_parser_requires_one_result_per_source():
    valid = {
        "sources": [
            {
                "id": 1,
                "supports_aspect": True,
                "confidence": 0.9,
                "evidence_role": "reputable_secondary",
            },
            {
                "id": 2,
                "supports_aspect": False,
                "confidence": 0.8,
                "evidence_role": "reject",
            },
        ]
    }

    parsed = ResearchConductor._parse_evidence_judgments(valid, 2)
    assert parsed is not None
    assert parsed[1]["evidence_role"] == "reputable_secondary"
    assert ResearchConductor._parse_evidence_judgments(
        {"sources": valid["sources"][:1]},
        2,
    ) is None


def make_judge_conductor():
    researcher = SimpleNamespace(
        query=(
            "How do I make style LoRAs or LoKRs for Krea 2 using "
            "Ostris AI Toolkit?"
        ),
        research_aspect=krea_aspect(),
        cfg=SimpleNamespace(
            fast_llm_model="fast-model",
            fast_llm_provider="openai",
            llm_kwargs={},
            source_evidence_judge_fallback="hybrid",
        ),
        add_costs=lambda _cost: None,
    )
    return ResearchConductor(researcher)


@pytest.mark.asyncio
async def test_all_source_judge_can_upgrade_reputable_but_not_first_party():
    conductor = make_judge_conductor()
    conductor._evidence_judge_excerpts = AsyncMock(
        return_value={0: ["official"], 1: ["established publication"], 2: ["blog"]}
    )
    sources = [
        {
            "url": "https://github.com/ostris/ai-toolkit",
            "title": "Ostris AI Toolkit",
            "raw_content": "Krea2 style LoRA training workflow",
            "_gptr_evidence_role": "first_party",
        },
        {
            "url": "https://publication.example/krea2",
            "title": "Krea 2 training analysis",
            "raw_content": "Krea2 Ostris ai-toolkit style LoRA training workflow",
            "_gptr_evidence_role": "practitioner",
        },
        {
            "url": "https://blog.example/krea2",
            "title": "Krea 2 tips",
            "raw_content": "Krea2 Ostris ai-toolkit style LoRA training workflow",
            "_gptr_evidence_role": "practitioner",
        },
    ]
    response = """{"sources":[
      {"id":1,"supports_aspect":true,"confidence":0.95,
       "evidence_role":"first_party","supported_entities":["Krea 2"],
       "supported_scope":[],"supported_claims":["workflow"],
       "corroborated_by":[],"reason":"official repository"},
      {"id":2,"supports_aspect":true,"confidence":0.9,
       "evidence_role":"reputable_secondary","supported_entities":["Krea 2"],
       "supported_scope":[],"supported_claims":["workflow"],
       "corroborated_by":[],"reason":"accountable publication"},
      {"id":3,"supports_aspect":true,"confidence":0.9,
       "evidence_role":"first_party","supported_entities":["Krea 2"],
       "supported_scope":[],"supported_claims":["workflow"],
       "corroborated_by":[],"reason":"self-described official"}
    ]}"""

    with patch(
        "gpt_researcher.skills.researcher.create_chat_completion",
        new=AsyncMock(return_value=response),
    ) as completion:
        accepted = await conductor._judge_evidence_sources(
            "Krea 2 Ostris AI Toolkit style LoRA training",
            sources,
        )

    assert completion.await_count == 1
    assert accepted == sources[:2]
    assert sources[1]["_gptr_evidence_role"] == "reputable_secondary"
    assert sources[2]["_gptr_evidence_role"] == "practitioner"


@pytest.mark.asyncio
async def test_malformed_judge_retries_once_then_uses_safe_hybrid():
    conductor = make_judge_conductor()
    conductor._evidence_judge_excerpts = AsyncMock(
        return_value={0: ["workflow"]}
    )
    source = {
        "url": "https://github.com/ostris/ai-toolkit",
        "title": "Ostris AI Toolkit Krea2 style LoRA training",
        "raw_content": (
            "Krea2 style LoRA and LoKR training workflow in Ostris "
            "AI Toolkit."
        ),
        "_gptr_evidence_role": "first_party",
    }

    with patch(
        "gpt_researcher.skills.researcher.create_chat_completion",
        new=AsyncMock(return_value="{malformed"),
    ) as completion:
        accepted = await conductor._judge_evidence_sources(
            "Krea 2 Ostris AI Toolkit style LoRA training",
            [source],
        )

    assert completion.await_count == 2
    assert accepted == [source]
    assert source["_gptr_judge_fallback"] is True


@pytest.mark.asyncio
async def test_rejected_source_cannot_corroborate_practitioner_claim():
    conductor = make_judge_conductor()
    conductor._evidence_judge_excerpts = AsyncMock(
        return_value={0: ["claim"], 1: ["different claim"]}
    )
    sources = [
        {
            "url": "https://one.example/krea",
            "title": "Krea workflow",
            "raw_content": "Krea2 Ostris ai-toolkit LoRA workflow",
            "_gptr_evidence_role": "practitioner",
        },
        {
            "url": "https://two.example/krea",
            "title": "Unrelated Krea page",
            "raw_content": "Krea2 overview",
            "_gptr_evidence_role": "practitioner",
        },
    ]
    response = """{"sources":[
      {"id":1,"supports_aspect":true,"confidence":0.9,
       "evidence_role":"practitioner","supported_claims":["LoRA workflow"],
       "corroborated_by":[2],"reason":"claims corroboration"},
      {"id":2,"supports_aspect":false,"confidence":0.9,
       "evidence_role":"practitioner","supported_claims":[],
       "corroborated_by":[],"reason":"does not support the workflow"}
    ]}"""

    with patch(
        "gpt_researcher.skills.researcher.create_chat_completion",
        new=AsyncMock(return_value=response),
    ):
        accepted = await conductor._judge_evidence_sources(
            "Krea 2 Ostris AI Toolkit style LoRA training",
            sources,
        )

    assert accepted == []


@pytest.mark.asyncio
async def test_preliminary_role_coverage_skips_redundant_serp():
    class MustNotSearch:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("strong preliminary evidence should be reused")

    class FakeEmbeddings:
        async def aembed_query(self, _query):
            return [1.0, 0.0]

        async def aembed_documents(self, _documents):
            return [[1.0, 0.0]]

    aspect = {
        **krea_aspect(),
        "evidence_roles": ["first_party"],
    }
    seed = {
        "url": "https://github.com/ostris/ai-toolkit",
        "title": "Ostris AI Toolkit Krea 2 training",
        "body": "Krea 2 style LoRA training workflow",
    }
    researcher = SimpleNamespace(
        retrievers=[MustNotSearch],
        cfg=SimpleNamespace(
            retrieval_pipeline_mode="v2",
            max_search_results_per_query=10,
            pre_scrape_source_curation=False,
        ),
        research_aspect=aspect,
        seed_candidates=[seed],
        candidate_ledger=EvidenceCandidateLedger(),
        research_policy=None,
        memory=SimpleNamespace(
            get_embeddings=lambda: FakeEmbeddings()
        ),
        verbose=False,
        websocket=None,
        visited_urls=set(),
        research_sources=[],
    )
    researcher.candidate_ledger.register(
        [seed],
        stage="preliminary",
        query="Krea 2",
    )
    conductor = ResearchConductor(researcher)

    urls, prefetched, _cards = (
        await conductor._search_relevant_source_urls(
            '"Krea 2" "Ostris AI Toolkit" style LoRA'
        )
    )

    assert urls == [seed["url"]]
    assert prefetched == []
    assert conductor.last_retrieval_diagnostics["search_executed"] is False


@pytest.mark.asyncio
async def test_failed_repository_page_recovers_raw_content_but_cites_canonical():
    class RepositoryRetriever:
        def __init__(self, _query, query_domains=None):
            self.query_domains = query_domains or []

        def search(self, max_results=10):
            return [
                {
                    "href": "https://github.com/ostris/ai-toolkit",
                    "title": "Ostris AI Toolkit Krea 2 training",
                    "body": "Krea 2 style LoRA training workflow",
                }
            ]

    class FakeEmbeddings:
        async def aembed_query(self, _query):
            return [1.0, 0.0]

        async def aembed_documents(self, documents):
            return [[1.0, 0.0] for _ in documents]

    scraper = SimpleNamespace(last_scrape_failures=[])

    async def browse_urls(urls, **_kwargs):
        if urls == ["https://github.com/ostris/ai-toolkit"]:
            return []
        main_readme = (
            "https://raw.githubusercontent.com/ostris/"
            "ai-toolkit/main/README.md"
        )
        return [
            {
                "url": main_readme,
                "title": "Ostris AI Toolkit Krea2 style LoRA training",
                "raw_content": (
                    "Krea2 style LoRA and LoKR training workflow using "
                    "Ostris AI Toolkit."
                ),
            }
        ] if main_readme in urls else []

    scraper.browse_urls = browse_urls
    researcher = SimpleNamespace(
        query="Krea 2 Ostris AI Toolkit style LoRA training",
        retrievers=[RepositoryRetriever],
        cfg=SimpleNamespace(
            retrieval_pipeline_mode="v2",
            max_search_results_per_query=10,
            pre_scrape_source_curation=False,
            post_scrape_source_integrity=True,
            canonical_content_resolution=True,
            source_evidence_judge_mode="off",
        ),
        research_aspect=krea_aspect(),
        seed_candidates=[],
        candidate_ledger=EvidenceCandidateLedger(),
        excluded_candidate_urls=set(),
        research_policy=None,
        memory=SimpleNamespace(
            get_embeddings=lambda: FakeEmbeddings()
        ),
        scraper_manager=scraper,
        vector_store=None,
        verbose=False,
        websocket=None,
        visited_urls=set(),
        research_sources=[],
    )
    researcher.add_research_sources = (
        lambda sources: researcher.research_sources.extend(sources)
    )
    conductor = ResearchConductor(researcher)

    recovered = await conductor._scrape_data_by_urls(
        researcher.query
    )

    assert len(recovered) == 1
    assert recovered[0]["url"] == "https://github.com/ostris/ai-toolkit"
    assert recovered[0]["_gptr_fetch_url"].startswith(
        "https://raw.githubusercontent.com/"
    )
    assert researcher.research_sources[0]["url"] == (
        "https://github.com/ostris/ai-toolkit"
    )
