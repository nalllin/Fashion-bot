"""Utilities for post-processing embeddings across services."""
from __future__ import annotations

from typing import Iterable, List


def compress_embedding(vector: Iterable[float], target_dim: int) -> List[float]:
    """Compress ``vector`` into ``target_dim`` dimensions via mean pooling."""

    vector_list = list(vector)
    if len(vector_list) == target_dim:
        return vector_list
    if target_dim <= 0:
        raise ValueError("target_dim must be greater than zero")
    chunk_size = len(vector_list) / target_dim
    compressed: List[float] = []
    for idx in range(target_dim):
        start = int(idx * chunk_size)
        end = int((idx + 1) * chunk_size)
        if start == end:
            end = min(start + 1, len(vector_list))
        if start >= len(vector_list):
            compressed.append(0.0)
            continue
        slice_values = vector_list[start:end]
        if not slice_values:
            compressed.append(0.0)
        else:
            compressed.append(float(sum(slice_values) / len(slice_values)))
    return compressed


def blend_embeddings(primary: Iterable[float], secondary: Iterable[float]) -> List[float]:
    """Average two vectors element-wise."""

    primary_list = list(primary)
    secondary_list = list(secondary)
    if len(primary_list) != len(secondary_list):
        raise ValueError("Embeddings must have the same dimension to blend")
    return [float((a + b) / 2) for a, b in zip(primary_list, secondary_list)]
