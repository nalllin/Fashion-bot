"""FAISS based vector store helper."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np


@dataclass
class VectorRecord:
    """Metadata stored for each embedding."""

    item_id: str
    payload: Dict[str, str]


@dataclass
class FaissVectorStore:
    """Simple FAISS wrapper with cosine similarity search."""

    dimension: int
    records: Dict[int, VectorRecord] = field(default_factory=dict)
    _vectors: List[List[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index = faiss.IndexFlatIP(self.dimension)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        return vectors / norms

    def add(self, embeddings: Iterable[Iterable[float]], records: Iterable[VectorRecord]) -> None:
        vectors = np.array(list(embeddings)).astype("float32")
        if vectors.size == 0:
            return
        if vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch for FAISS index")
        start = len(self.records)
        vectors = self._normalize(vectors)
        self._index.add(vectors)
        for offset, record in enumerate(records):
            self.records[start + offset] = record
        self._vectors.extend(vectors.tolist())

    def search(self, query: Iterable[float], k: int = 5) -> List[Tuple[VectorRecord, float]]:
        query_vector = np.array([list(query)], dtype="float32")
        query_vector = self._normalize(query_vector)
        scores, indices = self._index.search(query_vector, k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            record = self.records.get(index)
            if record:
                results.append((record, float(score)))
        return results

    def remove(self, item_id: str) -> None:
        indices_to_keep = [
            idx for idx, record in self.records.items() if record.item_id != item_id
        ]
        if len(indices_to_keep) == len(self.records):
            return
        new_vectors = []
        new_records: Dict[int, VectorRecord] = {}
        for new_idx, old_idx in enumerate(indices_to_keep):
            new_vectors.append(self._vectors[old_idx])
            new_records[new_idx] = self.records[old_idx]
        self._index.reset()
        if new_vectors:
            self._index.add(np.array(new_vectors, dtype="float32"))
        self.records = new_records
        self._vectors = new_vectors

    def get(self, item_id: str) -> Optional[VectorRecord]:
        for record in self.records.values():
            if record.item_id == item_id:
                return record
        return None
