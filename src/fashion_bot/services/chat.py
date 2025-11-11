"""Conversational stylist service."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..embeddings.text import TextEmbedder
from ..vectorstores.faiss_store import FaissVectorStore, VectorRecord
from .embedding_utils import compress_embedding

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


@dataclass
class ConversationMemory:
    """Simple fallback conversation memory when LangChain is unavailable."""

    history: List[Dict[str, str]] = field(default_factory=list)

    def add_user_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message})

    def add_ai_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})

    def get_context(self) -> List[Dict[str, str]]:
        return self.history[-10:]


@dataclass
class ChatStylist:
    """Interactive stylist chatbot leveraging GPT-4o-mini."""

    api_key: str
    wardrobe_store: FaissVectorStore
    text_embedder: TextEmbedder
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    model_name: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        if OpenAI is None:  # pragma: no cover - runtime guard
            raise ImportError("openai python package is required for ChatStylist")
        self._client = OpenAI(api_key=self.api_key)

    def _retrieve_context(self, query: str, k: int = 3) -> List[VectorRecord]:
        text_embedding = self.text_embedder.encode([query])[0]
        query_embedding = compress_embedding(text_embedding, self.wardrobe_store.dimension)
        results = self.wardrobe_store.search(query_embedding, k=k)
        return [record for record, _ in results]

    def chat(self, message: str) -> str:
        wardrobe_context = self._retrieve_context(message)
        context_blocks = [
            f"Item: {record.payload['category']} - {record.payload['description']}"
            for record in wardrobe_context
        ]
        context_text = "\n".join(context_blocks)
        history_before = list(self.memory.get_context())
        self.memory.add_user_message(message)
        conversation_history = "\n".join(
            f"{entry['role'].title()}: {entry['content']}"
            for entry in history_before
        )
        prompt = (
            "You are an AI fashion stylist. Use the provided wardrobe context to give "
            "personalised outfit advice.\n"
            f"Wardrobe Context:\n{context_text}\n"
            f"Conversation History:\n{conversation_history}\n"
            f"User: {message}"
        )
        response = self._client.responses.create(
            model=self.model_name,
            input=[{"role": "user", "content": prompt}],
        )
        reply = response.output[0].content[0].text  # type: ignore[index]
        self.memory.add_ai_message(reply)
        return reply
