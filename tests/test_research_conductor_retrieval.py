import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gpt_researcher.skills.researcher import ResearchConductor


class FakeSnippetRetriever:
    def __init__(self, query, query_domains=None):
        self.query = query
        self.query_domains = query_domains or []

    def search(self, max_results=10):
        return [
            {
                "href": "https://example.com/one",
                "body": "A" * 180,
            },
            {
                "href": "https://example.com/two",
                "body": "B" * 220,
            },
        ]


class FakeFullContentRetriever:
    def __init__(self, query, query_domains=None):
        self.query = query
        self.query_domains = query_domains or []

    def search(self, max_results=10):
        return [
            {
                "href": "https://example.com/full",
                "body": "short summary",
                "raw_content": "C" * 500,
            }
        ]


class ResearchConductorRetrievalTests(unittest.IsolatedAsyncioTestCase):
    def make_researcher(self, retriever_class):
        class FakeResearcher:
            def __init__(self):
                self.retrievers = [retriever_class]
                self.cfg = SimpleNamespace(
                    max_search_results_per_query=5,
                    pre_scrape_source_curation=False,
                )
                self.verbose = False
                self.websocket = None
                self.visited_urls = set()
                self.research_sources = []

            def add_research_sources(self, sources):
                self.research_sources.extend(sources)

        return FakeResearcher()

    async def test_snippet_only_results_are_sent_to_scraper(self):
        researcher = self.make_researcher(FakeSnippetRetriever)
        conductor = ResearchConductor(researcher)

        urls, prefetched, candidates = await conductor._search_relevant_source_urls(
            "rust async runtimes"
        )

        self.assertCountEqual(
            urls,
            ["https://example.com/one", "https://example.com/two"],
        )
        self.assertEqual(prefetched, [])
        self.assertEqual(set(candidates), set(urls))

    async def test_raw_content_results_stay_prefetched(self):
        researcher = self.make_researcher(FakeFullContentRetriever)
        conductor = ResearchConductor(researcher)

        urls, prefetched, candidates = await conductor._search_relevant_source_urls(
            "pubmed article"
        )

        self.assertEqual(urls, [])
        self.assertEqual(
            prefetched,
            [{
                "url": "https://example.com/full",
                "raw_content": "C" * 500,
                "title": "",
            }],
        )
        self.assertEqual(list(candidates), ["https://example.com/full"])

    async def test_only_selected_urls_are_passed_to_the_scraper(self):
        researcher = self.make_researcher(FakeSnippetRetriever)
        researcher.cfg.pre_scrape_source_curation = True
        researcher.scraper_manager = SimpleNamespace(calls=[])

        async def browse_urls(urls, **kwargs):
            researcher.scraper_manager.calls.append(urls)
            return []

        researcher.scraper_manager.browse_urls = browse_urls
        researcher.vector_store = None
        conductor = ResearchConductor(researcher)

        async def select_only_second(query, candidates):
            return [candidates[1]]

        conductor._select_source_candidates = select_only_second
        await conductor._scrape_data_by_urls("rust async runtimes")

        self.assertEqual(researcher.scraper_manager.calls, [["https://example.com/two"]])

    async def test_model_selected_off_topic_card_is_blocked_by_anchor_guard(self):
        researcher = self.make_researcher(FakeSnippetRetriever)
        researcher.cfg = SimpleNamespace(
            max_search_results_per_query=5,
            pre_scrape_source_curation=True,
            pre_scrape_max_sources_per_query=3,
            source_curation_policy="balanced",
            fast_llm_model="test-model",
            fast_llm_provider="openai",
            llm_kwargs={},
        )
        researcher.kwargs = {}
        researcher.add_costs = lambda _cost: None
        conductor = ResearchConductor(researcher)
        candidates = [
            {"href": "https://example.com/gnome", "title": "GNOME on Linux", "body": "Linux desktop fractional scaling"},
            {"href": "https://kernel.org/6.12", "title": "Linux 6.12 io_uring", "body": "io_uring async discard changes"},
        ]
        model_payload = '{"selected":[{"id":1,"reason":"bad choice"},{"id":2,"reason":"relevant"}]}'

        with patch("gpt_researcher.skills.researcher.create_chat_completion", new=AsyncMock(return_value=model_payload)), patch(
            "gpt_researcher.skills.researcher.stream_output", new=AsyncMock()
        ):
            selected = await conductor._select_source_candidates(
                "Linux kernel 6.12 io_uring changes", candidates
            )

        self.assertEqual([item["href"] for item in selected], ["https://kernel.org/6.12"])

    async def test_concurrent_branches_share_one_scrape_result(self):
        shared_visited = set()
        shared_lock = asyncio.Lock()
        shared_cache = {}
        shared_futures = {}
        scrape_calls = []

        def make_shared_researcher():
            researcher = self.make_researcher(FakeSnippetRetriever)
            researcher.visited_urls = shared_visited
            researcher.visited_urls_lock = shared_lock
            researcher.shared_scrape_cache = shared_cache
            researcher.shared_scrape_futures = shared_futures
            researcher.vector_store = None
            researcher.scraper_manager = SimpleNamespace()

            async def browse_urls(urls, **kwargs):
                if urls:
                    scrape_calls.extend(urls)
                    await asyncio.sleep(0.02)
                    return [
                        {
                            "url": url,
                            "title": "Rust async runtimes",
                            "raw_content": "Rust async runtime evidence " * 10,
                        }
                        for url in urls
                    ]
                return []

            researcher.scraper_manager.browse_urls = browse_urls
            return researcher

        first = ResearchConductor(make_shared_researcher())
        second = ResearchConductor(make_shared_researcher())
        first_result, second_result = await asyncio.gather(
            first._scrape_data_by_urls("rust async runtimes"),
            second._scrape_data_by_urls("rust async runtimes"),
        )

        self.assertEqual(
            sorted(scrape_calls),
            [
                "https://example.com/one",
                "https://example.com/two",
            ],
        )
        self.assertEqual(len(first_result), 2)
        self.assertEqual(len(second_result), 2)


if __name__ == "__main__":
    unittest.main()
