from types import SimpleNamespace
from datetime import datetime

import pytest
import asyncio
from collections import defaultdict
from unittest.mock import patch

import gpt_researcher.skills.deep_research as deep_research_module
from gpt_researcher.skills.deep_research import (
    DeepResearchSkill,
    MAX_CONTEXT_WORDS,
    parse_aspect_plan_response,
    parse_follow_up_questions_response,
    parse_research_results_response,
    parse_search_queries_response,
    _requested_recency_window,
    _research_subject_query,
    trim_context_to_word_limit,
)
from gpt_researcher.utils.research_budget import cold_start_policy


def test_aspect_plan_requires_distinct_original_query_anchors():
    response = """{"aspects":[
      {"id":"a","priority":1,"question":"How does Mage-Flow route models?","search_query":"Microsoft Mage-Flow model routing architecture","entities_versions_dates":["Mage-Flow"],"expected_evidence_type":"official docs","original_query_anchors":["Mage-Flow"]},
      {"id":"b","priority":2,"question":"What limitations does Mage-Flow have?","search_query":"Microsoft Mage-Flow limitations independent evaluation","entities_versions_dates":["Mage-Flow"],"expected_evidence_type":"technical evaluation","original_query_anchors":["Mage-Flow"]},
      {"id":"duplicate","priority":3,"question":"duplicate","search_query":"Microsoft Mage-Flow limitations independent evaluation","entities_versions_dates":[],"expected_evidence_type":"blog","original_query_anchors":[]},
      {"id":"drift","priority":4,"question":"unrelated","search_query":"banana bread recipe","entities_versions_dates":[],"expected_evidence_type":"recipe","original_query_anchors":[]}
    ]}"""

    plan = parse_aspect_plan_response(
        response, 4, "Research Microsoft Mage-Flow"
    )

    assert [item["id"] for item in plan] == ["a", "b"]


def test_aspect_plan_preserves_required_scope_anchors():
    response = """{"aspects":[{
      "id":"taxa","priority":1,
      "question":"Which predators affect major turtle groups?",
      "search_query":"turtle predators sea freshwater tortoise",
      "entities_versions_dates":[],
      "expected_evidence_type":"government or scholarly evidence",
      "original_query_anchors":["turtle","predators"],
      "required_scope_anchors":["sea turtles","freshwater turtles","tortoises"]
    }]}"""
    plan = parse_aspect_plan_response(
        response, 1, "What are the natural predators of turtles?"
    )
    assert plan[0]["required_scope_anchors"] == [
        "sea turtles",
        "freshwater turtles",
        "tortoises",
    ]


def test_single_fallback_domain_remains_a_gap_but_two_are_usable():
    skill = make_skill()
    skill.researcher.cfg.deep_research_fallback_corroboration_enabled = True
    skill.researcher.cfg.deep_research_fallback_corroboration = 2
    diagnostics = {
        "candidate_count": 2,
        "selected_count": 2,
        "scraped_count": 2,
        "accepted_count": 2,
    }

    state, details = skill._coverage_state(
        "evidence",
        [{"url": "https://one.example/article"}],
        diagnostics,
    )
    assert state == "corroboration_missing"
    assert details["corroborated"] is False

    state, details = skill._coverage_state(
        "evidence",
        [
            {"url": "https://one.example/article"},
            {"url": "https://two.example/article"},
        ],
        diagnostics,
    )
    assert state == "evidence_ready"
    assert details["corroborated"] is True


def test_partial_taxon_evidence_is_scope_missing():
    skill = make_skill()
    diagnostics = {
        "candidate_count": 1,
        "selected_count": 1,
        "scraped_count": 1,
        "accepted_count": 1,
    }
    state, details = skill._coverage_state(
        "Sea turtles have documented egg and hatchling predators.",
        [{"url": "https://www.fisheries.noaa.gov/species/sea-turtle"}],
        diagnostics,
        aspect={
            "required_scope_anchors": [
                "sea turtles",
                "freshwater turtles",
                "tortoises",
            ]
        },
    )
    assert state == "scope_missing"
    assert details["matched_scope_anchors"] == ["sea turtles"]
    assert details["missing_scope_anchors"] == [
        "freshwater turtles",
        "tortoises",
    ]


def test_repair_combines_partial_scope_instead_of_discarding_it():
    skill = make_skill()
    aspect = {
        "id": "taxa",
        "required_scope_anchors": [
            "freshwater turtles",
            "terrestrial turtles",
        ],
    }
    previous = {
        "aspect": aspect,
        "attempted_context": "Freshwater turtles face aquatic predators.",
        "attempted_sources": [
            {"url": "https://environment.nsw.gov.au/freshwater-turtles"}
        ],
        "retrieval_diagnostics": {
            "candidate_count": 1,
            "selected_count": 1,
            "scraped_count": 1,
            "accepted_count": 1,
        },
    }
    replacement = {
        "aspect": aspect,
        "attempted_context": "Terrestrial turtles face land predators.",
        "attempted_sources": [
            {"url": "https://example.museum/terrestrial-turtles"}
        ],
        "retrieval_diagnostics": {
            "candidate_count": 1,
            "selected_count": 1,
            "scraped_count": 1,
            "accepted_count": 1,
        },
    }

    combined = skill._combine_repair_evidence(previous, replacement)

    assert combined["coverage_state"] == "evidence_ready"
    assert len(combined["sources"]) == 2
    assert "Freshwater turtles" in combined["context"]
    assert "Terrestrial turtles" in combined["context"]


def test_aspect_query_encodes_expected_primary_evidence_standard():
    query = DeepResearchSkill._query_with_evidence_standard(
        {
            "search_query": "Mage-Flow routing architecture",
            "expected_evidence_type": (
                "official documentation or primary technical publication"
            ),
        }
    )

    assert "primary research paper" in query
    assert "official documentation" not in query


def test_aspect_plan_removes_meta_language_and_rejects_search_operators():
    response = """{"aspects":[
      {"id":"a","priority":1,"question":"How are embedding models trained?","search_query":"embedding model training original query","entities_versions_dates":[],"expected_evidence_type":"primary paper","original_query_anchors":["embedding"]},
      {"id":"b","priority":2,"question":"How are embedding models evaluated?","search_query":"site:example.com embedding model evaluation","entities_versions_dates":[],"expected_evidence_type":"benchmark paper","original_query_anchors":["embedding"]}
    ]}"""

    plan = parse_aspect_plan_response(response, 2, "Explain embedding models")

    assert [item["id"] for item in plan] == ["a"]
    assert plan[0]["search_query"] == "embedding model training"


def test_preliminary_search_subject_removes_request_framing():
    assert (
        _research_subject_query(
            "Please do deep research on Microsoft Mage-Flow architecture"
        )
        == "Microsoft Mage-Flow architecture"
    )
    assert (
        _research_subject_query("Explain embedding and retrieval models")
        == "embedding and retrieval models"
    )


def test_requested_recency_window_uses_full_date_interval():
    assert _requested_recency_window(
        "Find developments from the last 9 months",
        datetime(2026, 7, 23),
    ) == "2025-10-22 through 2026-07-23"


@pytest.mark.asyncio
async def test_fallback_corroboration_judge_can_reject_topical_similarity(
    monkeypatch,
):
    skill = make_skill()

    async def fake_completion(**_kwargs):
        return (
            '{"corroborated":false,'
            '"reason":"The pages discuss the topic but support different claims."}'
        )

    monkeypatch.setattr(
        deep_research_module, "create_chat_completion", fake_completion
    )
    corroborated, reason, mode = await skill._judge_fallback_corroboration(
        "Mage-Flow routing behavior",
        [
            {
                "url": "https://one.example/article",
                "raw_content": "Overview of Mage-Flow.",
            },
            {
                "url": "https://two.example/article",
                "raw_content": "Unrelated Mage-Flow benchmark.",
            },
        ],
    )

    assert corroborated is False
    assert "different claims" in reason
    assert mode == "llm"


@pytest.mark.asyncio
async def test_duplicate_repair_query_gets_failure_specific_variant(monkeypatch):
    skill = make_skill()
    skill.original_query = "Explain embedding and retrieval models"

    async def fake_completion(**_kwargs):
        return '{"query":"embedding retrieval architecture"}'

    monkeypatch.setattr(
        deep_research_module, "create_chat_completion", fake_completion
    )
    repaired = await skill.generate_repair_query(
        {
            "question": "What architecture do embedding retrieval models use?",
            "search_query": "embedding retrieval architecture",
        },
        "scrape_failure",
        {"attempted_queries": ["embedding retrieval architecture"]},
    )

    assert repaired.endswith("canonical PDF raw API documentation")


@pytest.mark.asyncio
async def test_missing_aspect_reuses_verified_primary_evidence(monkeypatch):
    skill = make_skill()

    async def fake_results(query, context, num_learnings=3):
        return {
            "learnings": [f"reused for {query}"],
            "followUpQuestions": [],
            "citations": {
                f"reused for {query}": (
                    "https://docs.vllm.ai/models/pooling_models/embed/"
                )
            },
        }

    monkeypatch.setattr(skill, "process_research_results", fake_results)
    source = {
        "url": "https://docs.vllm.ai/models/pooling_models/embed/",
        "title": "Embedding models",
        "raw_content": (
            "vLLM embedding pooling model runner configuration settings "
            "for production deployment"
        ),
    }
    results = [
        {
            "node_id": "root.1",
            "query": "vLLM embedding pooling model support",
            "coverage_state": "evidence_ready",
            "context": (
                "Official vLLM embedding pooling model runner configuration "
                "settings for production deployment."
            ),
            "sources": [source],
        },
        {
            "node_id": "root.2",
            "query": (
                "vLLM production configuration settings for embedding "
                "and pooling models"
            ),
            "aspect": {"id": "aspect-2"},
            "coverage_state": "integrity_failure",
            "context": "",
            "sources": [],
            "retrieval_diagnostics": {},
        },
    ]

    recovered = await skill._reuse_cross_aspect_evidence(
        results,
        node_id="root",
    )

    assert recovered[1]["coverage_state"] == "evidence_ready"
    assert recovered[1]["recovery_reason"] == (
        "reused verified cross-aspect evidence"
    )
    assert recovered[1]["sources"] == [source]
    assert recovered[1]["retrieval_diagnostics"]["cross_aspect_reuse"] is True


@pytest.mark.asyncio
async def test_cross_aspect_reuse_cannot_resolve_different_taxon_scope():
    skill = make_skill()
    sea_source = {
        "url": "https://www.seaturtlestatus.org/articles/predators",
        "title": "Sea turtle predators",
        "raw_content": "Sea turtles are preyed on by sharks.",
    }
    results = [
        {
            "node_id": "root.1",
            "query": "sea turtle predators",
            "coverage_state": "evidence_ready",
            "context": "Sea turtles are preyed on by sharks.",
            "sources": [sea_source],
        },
        {
            "node_id": "root.2",
            "query": "freshwater and terrestrial turtle predators",
            "aspect": {
                "id": "aspect-2",
                "required_scope_anchors": [
                    "freshwater turtles",
                    "terrestrial turtles",
                ],
            },
            "coverage_state": "scrape_failure",
            "context": "",
            "sources": [],
            "retrieval_diagnostics": {
                "candidate_count": 1,
                "selected_count": 1,
            },
        },
    ]

    recovered = await skill._reuse_cross_aspect_evidence(
        results,
        node_id="root",
    )

    assert recovered[1]["coverage_state"] == "scrape_failure"
    assert recovered[1]["sources"] == []


@pytest.mark.parametrize(
    ("diagnostics", "context", "sources", "expected"),
    [
        ({"candidate_count": 0}, "", [], "no_candidates"),
        (
            {"candidate_count": 2, "selected_count": 0},
            "",
            [],
            "no_qualified_source",
        ),
        (
            {"candidate_count": 2, "selected_count": 1, "scraped_count": 0},
            "",
            [],
            "scrape_failure",
        ),
        (
            {
                "candidate_count": 2,
                "selected_count": 1,
                "scraped_count": 1,
                "accepted_count": 0,
                "integrity_rejected_count": 1,
            },
            "",
            [],
            "integrity_failure",
        ),
        (
            {
                "candidate_count": 2,
                "selected_count": 1,
                "scraped_count": 1,
                "accepted_count": 1,
            },
            "",
            [{"url": "https://docs.example.com/a"}],
            "compression_empty",
        ),
    ],
)
def test_coverage_failure_states(diagnostics, context, sources, expected):
    skill = make_skill()
    state, _details = skill._coverage_state(context, sources, diagnostics)
    assert state == expected


def make_skill() -> DeepResearchSkill:
    cfg = SimpleNamespace(
        fast_llm_provider="openai",
        fast_llm_model="fast-model",
        strategic_llm_provider="anthropic",
        strategic_llm_model="claude-haiku-4-5",
        reasoning_effort="medium",
        llm_kwargs={},
        deep_research_breadth=3,
        deep_research_depth=2,
        deep_research_concurrency=2,
        deep_research_focused_retrieval=True,
        deep_research_max_deepened_branches=1,
        config_path=None,
    )
    researcher = SimpleNamespace(
        cfg=cfg,
        websocket=None,
        tone=None,
        headers={},
        visited_urls=set(),
        retrievers=[SimpleNamespace()],
        research_sources=[],
        mcp_configs=None,
        mcp_strategy=None,
    )
    return DeepResearchSkill(researcher)


@pytest.mark.asyncio
async def test_focused_three_by_two_runs_five_branches_and_only_deepens_best(monkeypatch):
    skill = make_skill()
    skill.concurrency_limit = 4
    skill._branch_semaphore = asyncio.Semaphore(4)
    created = []
    started = defaultdict(list)

    async def fake_queries(query, num_queries=3):
        if query == "topic":
            return [
                {"query": "alpha topic", "researchGoal": "alpha goal"},
                {"query": "beta topic", "researchGoal": "beta goal"},
                {"query": "gamma topic", "researchGoal": "gamma goal"},
            ]
        assert "beta goal" in query
        return [
            {"query": "beta detail one", "researchGoal": "beta detail one"},
            {"query": "beta detail two", "researchGoal": "beta detail two"},
        ]

    class FakeResearcher:
        def __init__(self, query, **kwargs):
            self.query = query
            self.kwargs = kwargs
            self.visited_urls = kwargs["visited_urls"]
            tier_url = "https://docs.example.test/beta" if "beta" in query else "https://blog.example.test/noise"
            self.research_sources = [{"url": tier_url, "title": query, "body": query}]
            created.append(self)

        async def conduct_research(self):
            started[self.query].append(asyncio.get_running_loop().time())
            await asyncio.sleep(0.02)
            return f"evidence for {self.query}"

    async def fake_results(query, context, num_learnings=3):
        return {
            "learnings": [f"learning {query}"],
            "followUpQuestions": [f"follow up {query}"],
            "citations": {f"learning {query}": f"https://citation.example/{query}"},
        }

    skill.generate_search_queries = fake_queries  # type: ignore[method-assign]
    skill.process_research_results = fake_results  # type: ignore[method-assign]
    with patch("gpt_researcher.GPTResearcher", FakeResearcher):
        result = await skill.deep_research("topic", breadth=3, depth=2)

    assert len(created) == 5
    assert all(item.kwargs["research_mode"] == "deep_branch" for item in created)
    assert skill._max_active_branches <= 4
    assert {item.query for item in created if "detail" in item.query} == {
        "topic beta detail one",
        "topic beta detail two",
    }
    assert len(result["sources"]) == 2
    assert len({source["url"] for source in result["sources"]}) == 2
    assert abs(
        started["topic beta detail one"][0]
        - started["topic beta detail two"][0]
    ) < 0.015


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tree_policy", "branch_mode", "expected_count", "expected_research_mode"),
    [
        ("legacy_all", "focused", 9, "deep_branch"),
        ("ranked", "standard", 5, "deep_branch_standard"),
    ],
)
async def test_tree_and_branch_policies_are_independent(
    tree_policy, branch_mode, expected_count, expected_research_mode
):
    skill = make_skill()
    skill.tree_policy = tree_policy
    skill.branch_mode = branch_mode
    created = []

    async def fake_queries(query, num_queries=3):
        if query == "topic":
            return [
                {
                    "query": f"{name} topic",
                    "researchGoal": f"{name} goal",
                }
                for name in ("alpha", "beta", "gamma")
            ]
        return [
            {"query": f"{query} detail {index}", "researchGoal": f"detail goal {index}"}
            for index in range(2)
        ]

    class FakeResearcher:
        def __init__(self, query, **kwargs):
            self.query = query
            self.kwargs = kwargs
            self.visited_urls = kwargs["visited_urls"]
            self.research_sources = [{
                "url": f"https://docs.example.test/{len(created)}",
                "title": query,
                "raw_content": f"evidence for {query}",
            }]
            created.append(self)

        async def conduct_research(self):
            return f"evidence for {self.query}"

    async def fake_results(query, context, num_learnings=3):
        return {
            "learnings": [f"learning {query}"],
            "followUpQuestions": [f"follow up {query}"],
            "citations": {},
        }

    skill.generate_search_queries = fake_queries  # type: ignore[method-assign]
    skill.process_research_results = fake_results  # type: ignore[method-assign]
    with patch("gpt_researcher.GPTResearcher", FakeResearcher):
        await skill.deep_research("topic", breadth=3, depth=2)

    assert len(created) == expected_count
    assert {
        researcher.kwargs["research_mode"] for researcher in created
    } == {expected_research_mode}


@pytest.mark.asyncio
async def test_ranked_tree_does_not_deepen_empty_evidence():
    skill = make_skill()
    created = []

    async def fake_queries(query, num_queries=3):
        assert query == "topic"
        return [
            {"query": f"empty aspect {index}", "researchGoal": f"goal {index}"}
            for index in range(3)
        ]

    class FakeResearcher:
        def __init__(self, query, **kwargs):
            self.query = query
            self.visited_urls = kwargs["visited_urls"]
            self.research_sources = []
            created.append(self)

        async def conduct_research(self):
            return ""

    async def fake_results(query, context, num_learnings=3):
        return {
            "learnings": [],
            "followUpQuestions": ["should not run"],
            "citations": {},
        }

    skill.generate_search_queries = fake_queries  # type: ignore[method-assign]
    skill.process_research_results = fake_results  # type: ignore[method-assign]
    with patch("gpt_researcher.GPTResearcher", FakeResearcher):
        await skill.deep_research("topic", breadth=3, depth=2)

    assert len(created) == 3


@pytest.mark.asyncio
async def test_thirty_second_policy_runs_two_aspects_and_no_children():
    skill = make_skill()
    skill.research_policy = cold_start_policy(30)
    skill.max_deepened_branches = 0
    created = []

    class FakeResearcher:
        def __init__(self, query, **kwargs):
            self.query = query
            self.visited_urls = kwargs["visited_urls"]
            self.research_sources = [
                {
                    "url": f"https://docs.example.test/{len(created)}",
                    "title": query,
                    "raw_content": f"verified evidence for {query}",
                }
            ]
            created.append(self)

        async def conduct_research(self):
            return f"verified evidence for {self.query}"

    async def fake_results(query, context, num_learnings=3):
        return {
            "learnings": [f"learning {query}"],
            "followUpQuestions": [f"follow up {query}"],
            "citations": {},
        }

    skill.process_research_results = fake_results  # type: ignore[method-assign]
    root_queries = [
        {
            "query": f"turtle predators aspect {index}",
            "researchGoal": f"goal {index}",
        }
        for index in range(2)
    ]
    with patch("gpt_researcher.GPTResearcher", FakeResearcher):
        await skill.deep_research(
            "turtle predators",
            breadth=2,
            depth=2,
            serp_queries_override=root_queries,
        )
    assert len(created) == 2
    assert skill._executed_work_units == 2


def stub_search_results(monkeypatch):
    async def fake_get_search_results(*args, **kwargs):
        return []

    monkeypatch.setattr(deep_research_module, "get_search_results", fake_get_search_results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            '[{"query":"q1","researchGoal":"g1"},{"query":"q2","researchGoal":"g2"}]',
            [
                {"query": "q1", "researchGoal": "g1"},
                {"query": "q2", "researchGoal": "g2"},
            ],
        ),
        (
            'Sure — here is the JSON.\n```json\n[{"query":"q1","researchGoal":"g1"}]\n```',
            [{"query": "q1", "researchGoal": "g1"}],
        ),
        (
            "1. Query: q1\n   Goal: g1\n2. Query: q2\n   Goal: g2",
            [
                {"query": "q1", "researchGoal": "g1"},
                {"query": "q2", "researchGoal": "g2"},
            ],
        ),
        (
            "- Query: q1\n- Goal: g1\n- Query: q2\n- Goal: g2",
            [
                {"query": "q1", "researchGoal": "g1"},
                {"query": "q2", "researchGoal": "g2"},
            ],
        ),
        (
            "Query: q1\nGoal: g1\nQuery: q2\nGoal: g2",
            [
                {"query": "q1", "researchGoal": "g1"},
                {"query": "q2", "researchGoal": "g2"},
            ],
        ),
    ],
)
async def test_generate_search_queries_parses_supported_formats(monkeypatch, response, expected):
    skill = make_skill()

    async def fake_create_chat_completion(**kwargs):
        return response

    monkeypatch.setattr(deep_research_module, "create_chat_completion", fake_create_chat_completion)

    assert await skill.generate_search_queries("topic", num_queries=3) == expected


@pytest.mark.asyncio
async def test_generate_search_queries_prompt_requires_json(monkeypatch):
    skill = make_skill()
    captured = {}

    async def fake_create_chat_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return "[]"

    monkeypatch.setattr(deep_research_module, "create_chat_completion", fake_create_chat_completion)

    await skill.generate_search_queries("topic", num_queries=2)

    system_prompt = captured["messages"][0]["content"]
    user_prompt = captured["messages"][1]["content"]
    assert "Return valid JSON only" in system_prompt
    assert "Return ONLY a JSON array of objects" in user_prompt


def test_parse_search_queries_response_repairs_trailing_comma():
    response = '[{"query": "test query", "researchGoal": "test goal",}]'

    result = parse_search_queries_response(response, num_queries=3)

    assert result
    assert result[0]["query"] == "test query"
    assert result[0]["researchGoal"] == "test goal"


def test_parse_search_queries_response_accepts_uppercase_json_fence():
    response = '```JSON\n[{"query": "x", "researchGoal": "y"}]\n```'

    result = parse_search_queries_response(response, num_queries=3)

    assert result
    assert result[0]["query"] == "x"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"questions":["What changed in 2025?","What should we compare?"]}', ["What changed in 2025?", "What should we compare?"]),
        ('Intro\n```json\n{"questions":["What changed in 2025?"]}\n```', ["What changed in 2025?"]),
        ("1. Question: What changed in 2025?\n2. Question: What should we compare?", ["What changed in 2025?", "What should we compare?"]),
        ("- What changed in 2025?\n- What should we compare?", ["What changed in 2025?", "What should we compare?"]),
    ],
)
async def test_generate_research_plan_parses_supported_formats(monkeypatch, response, expected):
    skill = make_skill()
    stub_search_results(monkeypatch)

    async def fake_create_chat_completion(**kwargs):
        return response

    monkeypatch.setattr(deep_research_module, "create_chat_completion", fake_create_chat_completion)

    assert await skill.generate_research_plan("topic", num_questions=3) == expected


@pytest.mark.asyncio
async def test_generate_research_plan_prompt_requires_json(monkeypatch):
    skill = make_skill()
    captured = {}
    stub_search_results(monkeypatch)

    async def fake_create_chat_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return '{"questions":[]}'

    monkeypatch.setattr(deep_research_module, "create_chat_completion", fake_create_chat_completion)

    await skill.generate_research_plan("topic", num_questions=2)

    system_prompt = captured["messages"][0]["content"]
    user_prompt = captured["messages"][1]["content"]
    assert "Return valid JSON only" in system_prompt
    assert '{"questions": ["<question 1>", "<question 2>"]}' in user_prompt


def test_parse_follow_up_questions_response_repairs_missing_quote():
    response = '{"questions": ["What is X", "Why is Y?]}'

    result = parse_follow_up_questions_response(response, num_questions=3)

    assert result
    assert result[0] == "What is X"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            '{"learnings":[{"insight":"Insight 1","sourceUrl":"https://a.test"},{"insight":"Insight 2","sourceUrl":""}],"followUpQuestions":["What next?","Why now?"]}',
            {
                "learnings": ["Insight 1", "Insight 2"],
                "followUpQuestions": ["What next?", "Why now?"],
                "citations": {"Insight 1": "https://a.test"},
            },
        ),
        (
            'Here is the JSON:\n```json\n{"learnings":[{"insight":"Insight 1","sourceUrl":"https://a.test"}],"followUpQuestions":["What next?"]}\n```',
            {
                "learnings": ["Insight 1"],
                "followUpQuestions": ["What next?"],
                "citations": {"Insight 1": "https://a.test"},
            },
        ),
        (
            "1. Learning [https://a.test]: Insight 1\n2. Learning [https://b.test]: Insight 2\n3. Question: What next?",
            {
                "learnings": ["Insight 1", "Insight 2"],
                "followUpQuestions": ["What next?"],
                "citations": {
                    "Insight 1": "https://a.test",
                    "Insight 2": "https://b.test",
                },
            },
        ),
        (
            "- Learning [https://a.test]: Insight 1\n- Learning: Insight 2 https://b.test\n- What next?",
            {
                "learnings": ["Insight 1", "Insight 2"],
                "followUpQuestions": ["What next?"],
                "citations": {
                    "Insight 1": "https://a.test",
                    "Insight 2": "https://b.test",
                },
            },
        ),
        (
            "Learning [https://a.test]: Insight 1\nQuestion: What next?",
            {
                "learnings": ["Insight 1"],
                "followUpQuestions": ["What next?"],
                "citations": {"Insight 1": "https://a.test"},
            },
        ),
    ],
)
async def test_process_research_results_parses_supported_formats(monkeypatch, response, expected):
    skill = make_skill()

    async def fake_create_chat_completion(**kwargs):
        return response

    monkeypatch.setattr(deep_research_module, "create_chat_completion", fake_create_chat_completion)

    assert await skill.process_research_results("topic", "context", num_learnings=3) == expected


@pytest.mark.asyncio
async def test_process_research_results_prompt_requires_json(monkeypatch):
    skill = make_skill()
    captured = {}

    async def fake_create_chat_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return '{"learnings":[],"followUpQuestions":[]}'

    monkeypatch.setattr(deep_research_module, "create_chat_completion", fake_create_chat_completion)

    await skill.process_research_results("topic", "context", num_learnings=2)

    system_prompt = captured["messages"][0]["content"]
    user_prompt = captured["messages"][1]["content"]
    assert "Return valid JSON only" in system_prompt
    assert '"learnings": [{"insight": "<insight>", "sourceUrl": "<url or empty string>"}]' in user_prompt


def test_parse_responses_return_empty_values_for_blank_input():
    expected_research_results = {
        "learnings": [],
        "followUpQuestions": [],
        "citations": {},
    }

    for response in ("", "   \n  \t  "):
        assert parse_search_queries_response(response, num_queries=3) == []
        assert parse_follow_up_questions_response(response, num_questions=3) == []
        assert parse_research_results_response(response, num_learnings=3) == expected_research_results


def test_parse_research_results_response_preserves_full_json_url():
    response = (
        '{"learnings": [{"insight": "fact",'
        ' "sourceUrl": "https://example.com/path?x=1&y=2"}],'
        ' "followUpQuestions": []}'
    )

    result = parse_research_results_response(response, num_learnings=3)

    assert result["citations"]["fact"] == "https://example.com/path?x=1&y=2"


def test_parse_research_results_response_extracts_inline_legacy_url():
    response = "Learning: stuff happened at https://example.com/api/v1?key=value"

    result = parse_research_results_response(response, num_learnings=3)

    assert result["learnings"] == ["stuff happened at"]
    assert result["citations"]["stuff happened at"] == "https://example.com/api/v1?key=value"


def test_trim_context_to_word_limit_keeps_first_item_when_it_alone_exceeds_cap():
    oversized_item = "word " * (MAX_CONTEXT_WORDS + 500)

    result = trim_context_to_word_limit([oversized_item], max_words=MAX_CONTEXT_WORDS)

    assert result, "trim_context_to_word_limit() should not return an empty context when the first item is oversized"
    assert len(result[0].split()) == MAX_CONTEXT_WORDS


def test_trim_context_to_word_limit_preserves_recent_context_and_keeps_one_oversized_tail_item():
    earlier = "early context " * 50
    oversized_latest = "latest " * (MAX_CONTEXT_WORDS + 250)

    result = trim_context_to_word_limit([earlier, oversized_latest], max_words=MAX_CONTEXT_WORDS)

    assert result == [" ".join(oversized_latest.split()[:MAX_CONTEXT_WORDS])]
