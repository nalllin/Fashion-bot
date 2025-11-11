"""Wardrobe management services."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from PIL import Image

from ..embeddings.image import ImageEmbedder
from ..embeddings.text import TextEmbedder
from ..vectorstores.faiss_store import FaissVectorStore, VectorRecord
from .embedding_utils import blend_embeddings, compress_embedding


@dataclass
class WardrobeItem:
    """Representation of a wardrobe item with metadata."""

    item_id: str
    category: str
    description: str
    image_path: str
    embedding_id: Optional[str] = None


@dataclass
class Wardrobe:
    """User wardrobe containing both metadata and embeddings."""

    user_id: str
    text_embedder: TextEmbedder
    image_embedder: ImageEmbedder
    vector_store: FaissVectorStore
    items: Dict[str, WardrobeItem] = field(default_factory=dict)

    def add_item(self, category: str, description: str, image: Image.Image) -> WardrobeItem:
        item_id = str(uuid.uuid4())
        image_embedding = self.image_embedder.encode([image])[0]
        text_embedding = self.text_embedder.encode([description])[0]
        compressed_text = compress_embedding(text_embedding, len(image_embedding))
        averaged_embedding = blend_embeddings(image_embedding, compressed_text)
        record = VectorRecord(
            item_id=item_id,
            payload={"category": category, "description": description},
        )
        self.vector_store.add([averaged_embedding], [record])
        wardrobe_item = WardrobeItem(
            item_id=item_id,
            category=category,
            description=description,
            image_path=f"wardrobe/{self.user_id}/{item_id}.png",
        )
        wardrobe_item.embedding_id = item_id
        self.items[item_id] = wardrobe_item
        return wardrobe_item

    def suggest_pairings(self, query_image: Image.Image, top_k: int = 5) -> List[WardrobeItem]:
        query_embedding = self.image_embedder.encode([query_image])[0]
        results = self.vector_store.search(query_embedding, k=top_k)
        return [self.items[result.item_id] for result, _ in results if result.item_id in self.items]

    def remove_item(self, item_id: str) -> None:
        if item_id not in self.items:
            return
        self.vector_store.remove(item_id)
        del self.items[item_id]


@dataclass
class WardrobeRepository:
    """In-memory repository that manages multiple wardrobes."""

    wardrobes: Dict[str, Wardrobe] = field(default_factory=dict)

    def get_or_create(
        self,
        user_id: str,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder,
        vector_store_factory: Callable[[], FaissVectorStore],
    ) -> Wardrobe:
        if user_id not in self.wardrobes:
            self.wardrobes[user_id] = Wardrobe(
                user_id=user_id,
                text_embedder=text_embedder,
                image_embedder=image_embedder,
                vector_store=vector_store_factory(),
            )
        return self.wardrobes[user_id]

    def list_items(self, user_id: str) -> Iterable[WardrobeItem]:
        wardrobe = self.wardrobes.get(user_id)
        if not wardrobe:
            return []
        return wardrobe.items.values()
