from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from ai_chat_stylist.chatbot import AIChatStylist, StylistConfig


class DummyRetriever:
    def __init__(self) -> None:
        self.docs: list[Document] = []

    def get_relevant_documents(self, _: str) -> list[Document]:
        return list(self.docs)


class DummyVectorStore:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.retriever = DummyRetriever()

    def add_texts(self, texts: list[str]) -> None:
        self.texts.extend(texts)
        for text in texts:
            self.retriever.docs.append(Document(page_content=text))

    def as_retriever(self, search_kwargs=None):
        return self.retriever


class FakeLLM:
    def __init__(self) -> None:
        self.last_prompt = None

    def invoke(self, messages):
        self.last_prompt = messages
        return AIMessage(content="Try the navy blazer with the striped tee.")


def build_stylist() -> AIChatStylist:
    vectorstore = DummyVectorStore()
    stylist = AIChatStylist(
        config=StylistConfig(),
        llm_factory=lambda cfg: FakeLLM(),
        embedding_factory=lambda cfg: None,
        vectorstore=vectorstore,
    )
    stylist.vectorstore = vectorstore
    stylist.retriever = vectorstore.as_retriever()
    return stylist


def test_ingest_text_adds_to_vectorstore():
    stylist = build_stylist()
    stylist.ingest_text("Capsule wardrobe basics include a denim jacket.")
    assert any(
        "denim jacket" in text for text in stylist.vectorstore.texts
    ), "Context text should be added to the FAISS store."


def test_add_outfit_image_ingests_description():
    stylist = build_stylist()
    with patch(
        "ai_chat_stylist.chatbot.describe_outfit_image",
        return_value="A navy blazer with tailored trousers.",
    ):
        description = stylist.add_outfit_image("fake.png")
    assert "navy blazer" in description
    assert any("navy blazer" in text for text in stylist.vectorstore.texts)


def test_chat_raises_on_empty_input():
    stylist = build_stylist()
    with pytest.raises(ValueError):
        stylist.chat("   ")


def test_chat_appends_history_and_vector_memory():
    stylist = build_stylist()
    reply = stylist.chat("What can I wear with my navy blazer?")
    assert "navy blazer" in stylist.vectorstore.texts[-2]
    assert reply.startswith("Try the navy blazer")
