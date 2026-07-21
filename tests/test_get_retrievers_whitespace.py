"""Regression tests for whitespace handling in get_retrievers.

Comma-separated retriever lists supplied via request headers
(e.g. ``"tavily, exa"``) previously kept the surrounding whitespace,
so every name after the first failed the exact ``match`` in
``get_retriever`` and silently fell back to the default retriever.
The config path already stripped whitespace; these tests pin the same
behavior for the header path (and for a single-retriever value).
"""

import unittest
from types import SimpleNamespace

from gpt_researcher.config.config import Config
from gpt_researcher.actions.retriever import (
    get_retrievers,
    get_retriever,
    get_default_retriever,
)


def _cfg(retrievers=None, retriever=None):
    return SimpleNamespace(retrievers=retrievers, retriever=retriever)


class GetRetrieversWhitespaceTests(unittest.TestCase):
    def test_header_comma_list_with_spaces_resolves_each_retriever(self):
        # "tavily, exa" -> the second name has a leading space.
        headers = {"retrievers": "tavily, exa"}
        result = get_retrievers(headers, _cfg())

        expected = [get_retriever("tavily"), get_retriever("exa")]
        self.assertEqual(result, expected)

        # The buggy behavior silently substituted the default retriever for
        # the un-stripped " exa"; assert that did NOT happen.
        default = get_default_retriever()
        self.assertEqual(result[1], get_retriever("exa"))
        self.assertNotEqual(
            result[1].__name__ if result[1] else None,
            None,
        )
        # exa is not the default, so a correct parse must differ from default
        self.assertNotEqual(get_retriever("exa"), default)

    def test_single_header_retriever_with_spaces_is_stripped(self):
        headers = {"retriever": "  tavily  "}
        result = get_retrievers(headers, _cfg())
        self.assertEqual(result, [get_retriever("tavily")])

    def test_blank_entries_are_dropped(self):
        headers = {"retrievers": "tavily, , exa,"}
        result = get_retrievers(headers, _cfg())
        self.assertEqual(result, [get_retriever("tavily"), get_retriever("exa")])

    def test_config_string_path_still_strips(self):
        result = get_retrievers({}, _cfg(retrievers="tavily, exa"))
        self.assertEqual(result, [get_retriever("tavily"), get_retriever("exa")])

    def test_searxng_alias_resolves_to_searx(self):
        self.assertEqual(get_retriever("searxng"), get_retriever("searx"))
        result = get_retrievers({}, _cfg(retrievers="searxng"))
        self.assertEqual(result, [get_retriever("searx")])

    def test_config_parser_normalizes_searxng_alias(self):
        cfg = Config.__new__(Config)
        self.assertEqual(cfg.parse_retrievers("searxng"), ["searx"])


if __name__ == "__main__":
    unittest.main()
