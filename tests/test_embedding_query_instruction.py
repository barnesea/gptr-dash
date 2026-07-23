import asyncio
from unittest.mock import patch

from gpt_researcher.memory.embeddings import Memory, QueryInstructionEmbeddings


class FakeEmbeddings:
    def __init__(self):
        self.document_inputs = []
        self.query_inputs = []

    def embed_documents(self, texts):
        self.document_inputs.append(texts)
        return [[float(len(text))] for text in texts]

    def embed_query(self, text):
        self.query_inputs.append(text)
        return [float(len(text))]

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


def test_query_instruction_is_not_applied_to_documents():
    base = FakeEmbeddings()
    embeddings = QueryInstructionEmbeddings(
        base, "Given a web search query, retrieve relevant passages that answer the query"
    )

    embeddings.embed_documents(["A source passage"])
    embeddings.embed_query("How do embeddings work?")
    asyncio.run(embeddings.aembed_query("What is retrieval?"))

    assert base.document_inputs == [["A source passage"]]
    assert base.query_inputs == [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery: How do embeddings work?",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery: What is retrieval?",
    ]


def test_native_query_and_document_prefixes_are_applied_asymmetrically():
    base = FakeEmbeddings()
    embeddings = QueryInstructionEmbeddings(
        base, query_prefix="Query: ", document_prefix="Document: "
    )

    embeddings.embed_documents(["A source passage"])
    embeddings.embed_query("How do embeddings work?")

    assert base.document_inputs == [["Document: A source passage"]]
    assert base.query_inputs == ["Query: How do embeddings work?"]


def test_openai_embedding_base_url_can_differ_from_chat_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://chat.example/v1")
    monkeypatch.setenv("EMBEDDING_OPENAI_BASE_URL", "http://embed.example/v1")
    with patch("langchain_openai.OpenAIEmbeddings") as embedding_class:
        Memory("openai", "jina-v5-retrieval")

    assert embedding_class.call_args.kwargs["openai_api_base"] == "http://embed.example/v1"
