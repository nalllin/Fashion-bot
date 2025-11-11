"""Text embedding utilities for the Fashion Bot platform."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


class EmbeddingBackend(Protocol):
    """Minimal protocol implemented by embedding backends."""

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        """Return embeddings for ``texts``."""


@dataclass
class SentenceTransformerBackend:
    """Embedding backend that wraps ``SentenceTransformer`` models."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        if SentenceTransformer is None:  # pragma: no cover - runtime guard
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerBackend"
            )
        self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return self._model.encode(list(texts), convert_to_numpy=False)  # type: ignore[arg-type]


@dataclass
class OpenAIEmbeddingBackend:
    """Embedding backend that uses OpenAI's ``text-embedding-3-small`` model."""

    api_key: str
    model_name: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        if OpenAI is None:  # pragma: no cover - runtime guard
            raise ImportError("openai python package is required for OpenAIEmbeddingBackend")
        self._client = OpenAI(api_key=self.api_key)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        result = self._client.embeddings.create(model=self.model_name, input=list(texts))
        return [item.embedding for item in result.data]


@dataclass
class TextEmbedder:
    """Convenience wrapper that delegates to an ``EmbeddingBackend``."""

    backend: EmbeddingBackend

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        """Return vector representations for ``texts``."""

        return self.backend.embed_documents(texts)
