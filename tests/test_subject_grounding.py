import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gpt_researcher.skills.deep_research as deep_research_module
from gpt_researcher.skills.deep_research import (
    DeepResearchSkill,
    _fallback_subject_grounding_plan,
    parse_subject_grounding_plan_response,
    parse_subject_grounding_response,
)
from gpt_researcher.utils.llm import create_chat_completion
from gpt_researcher.utils.subject_grounding import (
    get_subject_grounding,
    subject_grounding_context,
    subject_grounding_instruction,
)


def _make_grounding_skill() -> DeepResearchSkill:
    cfg = SimpleNamespace(
        fast_llm_provider="openai",
        fast_llm_model="laguna:gptr",
        strategic_llm_provider="openai",
        strategic_llm_model="laguna:gptr",
        reasoning_effort="medium",
        llm_kwargs={},
        deep_research_breadth=2,
        deep_research_depth=1,
        deep_research_concurrency=3,
        deep_research_focused_retrieval=True,
        deep_research_max_deepened_branches=0,
        retrieval_pipeline_mode="v2",
        planning_search_results=8,
        subject_grounding_enabled=True,
        subject_grounding_max_subjects=4,
        subject_grounding_results_per_query=3,
        subject_grounding_search_concurrency=3,
        config_path=None,
    )
    researcher = SimpleNamespace(
        cfg=cfg,
        query=(
            "How do I make style LoRAs or LoKRs with Krea 2 using "
            "Ostris AI Toolkit?"
        ),
        websocket=None,
        tone=None,
        headers={},
        visited_urls=set(),
        retrievers=[SimpleNamespace(__name__="SearxSearch")],
        research_sources=[],
        mcp_configs=None,
        mcp_strategy=None,
        subject_grounding={},
    )
    return DeepResearchSkill(researcher)


def test_subject_grounding_plan_is_positive_and_dependency_ordered():
    response = """{"subjects":[
      {"id":"subject-1","name":"Krea 2","role":"main",
       "definition_question":"What is Krea 2?",
       "search_query":"what is Krea 2","depends_on":[]},
      {"id":"subject-2","name":"LoRA","role":"supporting",
       "definition_question":"What is a LoRA for image models?",
       "search_query":"LoRA for image generation models",
       "depends_on":["subject-1"]},
      {"id":"subject-3","name":"LoKr","role":"supporting",
       "definition_question":"What is a LoKr for image models?",
       "search_query":"LoKr for image generation models",
       "depends_on":["missing","subject-1"]}
    ],"excluded_topics":["Krea 1"]}"""

    plan = parse_subject_grounding_plan_response(
        response,
        4,
        "How do I train Krea 2 LoRA and LoKr adapters?",
    )

    assert [item["name"] for item in plan] == ["Krea 2", "LoRA", "LoKr"]
    assert plan[0]["role"] == "main"
    assert plan[0]["depends_on"] == []
    assert plan[2]["depends_on"] == ["subject-1"]
    assert all("excluded" not in item for item in plan)


def test_invalid_subject_plan_fallback_extracts_literal_topic_hierarchy():
    plan = _fallback_subject_grounding_plan(
        (
            "How do I make style LoRAs or LoKRs with Krea 2 using "
            "Ostris AI Toolkit? Cover training and validation."
        ),
        4,
    )

    assert plan[0]["name"] == "Krea 2"
    assert plan[0]["role"] == "main"
    assert [item["name"] for item in plan[1:]] == [
        "style LoRAs",
        "LoKRs",
        "Ostris AI Toolkit",
    ]
    assert all(
        item["depends_on"] == ["subject-1"]
        for item in plan[1:]
    )


def test_subject_grounding_response_extracts_only_orientation_fields():
    payload = parse_subject_grounding_response(
        """{"grounding_statement":
        "Krea 2 is an image generation model family. LoRA and LoKr are adapter
        forms used to specialize a base model, while Ostris AI Toolkit is the
        training software named in this question. The research task is to
        establish how those adapters are trained for Krea 2 with that toolkit.",
        "subject_definitions":[
          {"name":"Krea 2","definition":"An image generation model family."}
        ],
        "relationship":"adapter training with the named toolkit",
        "excluded_topics":["another product"]}"""
    )

    assert payload["statement"].startswith("Krea 2")
    assert payload["subject_definitions"][0]["name"] == "Krea 2"
    assert "excluded_topics" not in payload


def test_grounding_preserves_general_positive_variants_with_source_labels():
    payload = parse_subject_grounding_response(
        """{
          "grounding_statement":"AtlasDB is the named database family in this
          research question. AtlasDB Core and AtlasDB Edge are positively
          identified variants with different stated roles, and the requested
          comparison must keep each observation attached to its variant.",
          "subject_definitions":[
            {"name":"AtlasDB","definition":"A database product family."}
          ],
          "defining_facts":[
            {"subject":"AtlasDB Core",
             "fact":"The general-purpose member of the AtlasDB family.",
             "fact_type":"variant","confidence_label":"high",
             "evidence_urls":["https://docs.example.test/atlasdb"]},
            {"subject":"AtlasDB Edge",
             "fact":"A variant intended for edge deployments.",
             "fact_type":"role","confidence_label":"medium",
             "evidence_urls":["https://docs.example.test/atlasdb-edge"]}
          ],
          "relationship":"compare the two named variants"
        }"""
    )

    assert [fact["subject"] for fact in payload["defining_facts"]] == [
        "AtlasDB Core",
        "AtlasDB Edge",
    ]
    instruction = subject_grounding_instruction(payload)
    assert "[high; variant] AtlasDB Core" in instruction
    assert "[medium; role] AtlasDB Edge" in instruction
    assert "not final report evidence" in instruction


@pytest.mark.asyncio
async def test_llm_grounding_is_request_scoped_and_does_not_mutate_messages():
    provider = MagicMock()
    captured = []

    async def respond(messages, *_args, **_kwargs):
        captured.append(messages)
        return "ok"

    provider.get_chat_response = AsyncMock(side_effect=respond)
    original = [{"role": "system", "content": "Return JSON."}]
    grounding = {
        "statement": (
            "The main subject is Alpha, and Beta is the supporting method "
            "whose relationship to Alpha is being researched."
        )
    }

    with patch(
        "gpt_researcher.utils.llm.get_llm",
        return_value=provider,
    ):
        with subject_grounding_context(grounding):
            assert get_subject_grounding() == grounding
            await create_chat_completion(
                messages=original,
                model="laguna:gptr",
                llm_provider="openai",
            )
        assert get_subject_grounding() is None
        await create_chat_completion(
            messages=original,
            model="laguna:gptr",
            llm_provider="openai",
        )

    assert "SUBJECT GROUNDING" in captured[0][0]["content"]
    assert "Alpha" in captured[0][0]["content"]
    assert "SUBJECT GROUNDING" not in captured[1][0]["content"]
    assert original == [{"role": "system", "content": "Return JSON."}]


@pytest.mark.asyncio
async def test_grounding_searches_dependent_subjects_in_parallel(monkeypatch):
    skill = _make_grounding_skill()
    active = 0
    max_active = 0
    searches = []

    async def fake_search(query, *_args, **_kwargs):
        nonlocal active, max_active
        searches.append(query)
        if query.startswith("How do I make"):
            return []
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return [
            {
                "title": f"Definition for {query}",
                "href": f"https://example.test/{len(searches)}",
                "body": f"Evidence describing {query}.",
            }
        ]

    completions = iter(
        [
            """{"subjects":[
              {"id":"subject-1","name":"Krea 2","role":"main",
               "definition_question":"What is Krea 2?",
               "search_query":"Krea 2 image model","depends_on":[]},
              {"id":"subject-2","name":"LoRA","role":"supporting",
               "definition_question":"What is LoRA here?",
               "search_query":"LoRA adapter","depends_on":["subject-1"]},
              {"id":"subject-3","name":"LoKr","role":"supporting",
               "definition_question":"What is LoKr here?",
               "search_query":"LoKr adapter","depends_on":["subject-1"]}
            ]}""",
            """{"grounding_statement":"Krea 2 is the image model family in this
            research question. LoRA and LoKr are adapter forms used to
            specialize a base image model, and Ostris AI Toolkit is the named
            training software. The research must establish the practical
            relationship among Krea 2, each adapter form, and that toolkit.",
            "subject_definitions":[],
            "relationship":"training Krea 2 adapters with Ostris AI Toolkit",
            "aspects":[
              {"id":"a1","priority":1,"intent":"procedural",
               "question":"How is a Krea 2 style LoRA trained with Ostris AI Toolkit?",
               "search_query":"Krea 2 Ostris AI Toolkit style LoRA training",
               "entities":[
                 {"name":"Krea 2","aliases":[],"exact":true},
                 {"name":"Ostris AI Toolkit","aliases":["ai-toolkit"],"exact":true},
                 {"name":"LoRA","aliases":[],"exact":true}
               ],
               "relation":"style LoRA training workflow",
               "evidence_roles":["first_party","practitioner"],
               "expected_evidence_type":"first-party and practitioner workflow",
               "original_query_anchors":["Krea 2"],"required_scope_anchors":[]},
              {"id":"a2","priority":2,"intent":"procedural",
               "question":"How does LoKr training differ for Krea 2?",
               "search_query":"Krea 2 LoKr Ostris AI Toolkit training differences",
               "entities":[
                 {"name":"Krea 2","aliases":[],"exact":true},
                 {"name":"Ostris AI Toolkit","aliases":["ai-toolkit"],"exact":true},
                 {"name":"LoKr","aliases":[],"exact":true}
               ],
               "relation":"LoKr training differences",
               "evidence_roles":["first_party","practitioner"],
               "expected_evidence_type":"practitioner configuration evidence",
               "original_query_anchors":["Krea 2"],"required_scope_anchors":[]}
            ]}""",
        ]
    )

    async def fake_completion(**_kwargs):
        return next(completions)

    monkeypatch.setattr(
        deep_research_module,
        "get_search_results",
        fake_search,
    )
    monkeypatch.setattr(
        deep_research_module,
        "create_chat_completion",
        fake_completion,
    )

    plan = await skill.generate_aspect_plan(
        skill.researcher.query,
        2,
    )

    assert len(plan) == 2
    assert max_active == 3
    assert any(
        query == "LoRA adapter in the context of Krea 2"
        for query in searches
    )
    assert any(
        query == "LoKr adapter in the context of Krea 2"
        for query in searches
    )
    assert "Krea 2 is the image model family" in (
        skill.researcher.subject_grounding["statement"]
    )
    assert plan[0]["required_scope_anchors"] == [
        "Krea 2",
        "Ostris AI Toolkit",
        "LoRA",
    ]
    assert plan[1]["required_scope_anchors"] == [
        "Krea 2",
        "Ostris AI Toolkit",
        "LoKr",
    ]


@pytest.mark.asyncio
async def test_cross_aspect_reuse_does_not_relabel_a_prior_judgment():
    skill = _make_grounding_skill()
    source = {
        "url": "https://docs.example.test/krea",
        "title": "Krea 2 Ostris AI Toolkit LoRA training",
        "raw_content": (
            "Krea 2 Ostris AI Toolkit LoRA loading and inference settings"
        ),
        "_gptr_evidence_judgment": {
            "aspect_id": "configuration",
            "supports_aspect": True,
            "fallback": "",
        },
    }
    results = [
        {
            "node_id": "root.1",
            "query": "Krea 2 Ostris AI Toolkit LoRA configuration",
            "aspect": {"id": "configuration"},
            "coverage_state": "evidence_ready",
            "context": source["raw_content"],
            "sources": [source],
        },
        {
            "node_id": "root.2",
            "query": "Krea 2 Ostris AI Toolkit LoRA dataset captioning",
            "aspect": {"id": "dataset"},
            "coverage_state": "no_qualified_source",
            "context": "",
            "sources": [],
        },
    ]

    recovered = await skill._reuse_cross_aspect_evidence(
        results,
        node_id="root",
    )

    assert recovered[1]["coverage_state"] == "no_qualified_source"
    assert recovered[1]["sources"] == []


def test_background_evidence_is_kept_but_does_not_resolve_relationship():
    skill = _make_grounding_skill()
    skill.original_query = (
        "How do I train Krea 2 with Ostris AI Toolkit?"
    )
    source = {
        "url": "https://github.com/krea-ai/krea-2",
        "title": "Krea 2",
        "raw_content": (
            "Krea 2 recommends Ostris AI Toolkit as a finetuning tool."
        ),
        "_gptr_evidence_judgment": {
            "aspect_id": "workflow",
            "supports_aspect": True,
            "evidence_application": "background",
            "confidence": 0.8,
            "supported_entities": ["Krea 2", "Ostris AI Toolkit"],
            "supported_scope": [],
            "supported_claims": [
                "Krea 2 recommends Ostris AI Toolkit for finetuning."
            ],
            "reason": "The page recommends the tool but has no workflow.",
            "fallback": "",
        },
    }
    diagnostics = {
        "candidate_count": 1,
        "selected_count": 1,
        "scraped_count": 1,
        "accepted_count": 1,
    }
    aspect = {
        "id": "workflow",
        "question": "How is Krea 2 trained with Ostris AI Toolkit?",
        "search_query": "Krea 2 Ostris AI Toolkit training workflow",
        "relation": "training workflow",
        "entities": [
            {"name": "Krea 2", "aliases": [], "exact": True},
            {
                "name": "Ostris AI Toolkit",
                "aliases": [],
                "exact": True,
            },
        ],
    }

    state, details = skill._coverage_state(
        source["raw_content"],
        [source],
        diagnostics,
        aspect=aspect,
    )
    ledger = skill._coverage_ledger_entry(
        aspect,
        {
            "query": aspect["search_query"],
            "coverage_state": state,
            "sources": [],
            "attempted_sources": [source],
            "retrieval_diagnostics": diagnostics,
            **details,
        },
    )

    assert state == "compression_empty"
    assert details["background_evidence_only"] is True
    assert ledger["verified_urls"] == []
    assert ledger["background_urls"] == [source["url"]]
    assert (
        ledger["background_sources"][0]["evidence_application"]
        == "background"
    )


def test_exact_entity_scope_cannot_be_satisfied_by_shared_words():
    skill = _make_grounding_skill()
    skill.original_query = (
        "How do I train Krea 2 LoRA and LoKr adapters with "
        "Ostris AI Toolkit?"
    )
    source = {
        "url": "https://krea.ai/blog/krea-2-lora-training",
        "title": "Krea 2 LoRA training",
        "_gptr_source_tier": "primary",
        "_gptr_evidence_judgment": {
            "aspect_id": "workflow",
            "supports_aspect": True,
            "accepted_for_synthesis": True,
            "evidence_application": "direct",
            "confidence": 0.95,
            "supported_entities": ["Krea 2", "LoRA"],
            "supported_scope": [],
            "supported_claims": [
                "Krea 2 LoRA training uses a curated image dataset."
            ],
            "reason": "The page covers Krea 2 LoRA training only.",
            "fallback": "",
        },
    }
    aspect = {
        "id": "workflow",
        "question": (
            "How are Krea 2 LoRA and LoKr adapters trained with "
            "Ostris AI Toolkit?"
        ),
        "search_query": (
            "Krea 2 LoRA LoKr Ostris AI Toolkit training"
        ),
        "relation": "adapter training workflow",
        "entities": [
            {"name": "Krea 2", "aliases": [], "exact": True},
            {
                "name": "Ostris AI Toolkit",
                "aliases": ["ai-toolkit"],
                "exact": True,
            },
            {"name": "LoRA", "aliases": [], "exact": True},
            {"name": "LoKr", "aliases": [], "exact": True},
        ],
        "required_scope_anchors": [
            "Krea 2",
            "Ostris AI Toolkit",
            "LoRA",
            "LoKr",
        ],
    }
    diagnostics = {
        "candidate_count": 1,
        "selected_count": 1,
        "scraped_count": 1,
        "accepted_count": 1,
    }

    state, details = skill._coverage_state(
        (
            "Source: https://krea.ai/blog/krea-2-lora-training\n"
            "Title: Krea 2 LoRA training\n"
            "Content: Krea 2 LoRA training uses a curated image dataset."
        ),
        [source],
        diagnostics,
        aspect=aspect,
    )

    assert state == "scope_missing"
    assert details["matched_scope_anchors"] == ["Krea 2", "LoRA"]
    assert details["missing_scope_anchors"] == [
        "Ostris AI Toolkit",
        "LoKr",
    ]


def test_partial_scope_excludes_background_text_from_synthesis():
    skill = _make_grounding_skill()
    skill.original_query = (
        "How do I train Krea 2 with Ostris AI Toolkit?"
    )
    direct = {
        "url": "https://krea.ai/blog/krea-2-lora-training",
        "title": "Krea 2 LoRA training",
        "_gptr_source_tier": "primary",
        "_gptr_evidence_judgment": {
            "aspect_id": "workflow",
            "supports_aspect": True,
            "accepted_for_synthesis": True,
            "evidence_application": "direct",
            "supported_entities": ["Krea 2"],
            "supported_scope": ["Krea 2"],
            "fallback": "",
        },
    }
    background = {
        "url": "https://example.test/ostris-settings",
        "title": "Unverified toolkit settings",
        "_gptr_source_tier": "fallback",
        "_gptr_evidence_judgment": {
            "aspect_id": "workflow",
            "supports_aspect": True,
            "accepted_for_synthesis": False,
            "evidence_application": "background",
            "supported_entities": ["Ostris AI Toolkit"],
            "supported_scope": ["Ostris AI Toolkit"],
            "fallback": "",
        },
    }
    result = {
        "node_id": "root.1",
        "query": "Krea 2 Ostris AI Toolkit training",
        "aspect": {
            "id": "workflow",
            "question": "Krea 2 and Ostris AI Toolkit training",
            "relation": "training workflow",
            "required_scope_anchors": [
                "Krea 2",
                "Ostris AI Toolkit",
            ],
        },
        "coverage_state": "scope_missing",
        "matched_scope_anchors": ["Krea 2"],
        "missing_scope_anchors": ["Ostris AI Toolkit"],
        "attempted_context": (
            "Source: https://krea.ai/blog/krea-2-lora-training\n"
            "Title: Krea 2 LoRA training\n"
            "Content: Verified Krea 2 dataset guidance.\n"
            "Source: https://example.test/ostris-settings\n"
            "Title: Unverified toolkit settings\n"
            "Content: Unsupported optimizer and resolution claims."
        ),
        "attempted_sources": [direct, background],
        "retrieval_diagnostics": {
            "candidate_count": 2,
            "selected_count": 2,
            "scraped_count": 2,
            "accepted_count": 2,
        },
    }

    partial = skill._partial_scope_result(result)

    assert partial is not None
    assert partial["sources"] == [direct]
    assert "Verified Krea 2 dataset guidance" in partial["context"]
    assert "Unsupported optimizer" not in partial["context"]
