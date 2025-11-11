"""Image embedding utilities leveraging CLIP or OpenAI vision models."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List, Protocol

from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    from transformers import CLIPModel, CLIPProcessor
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    T = None  # type: ignore
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


class ImageEmbeddingBackend(Protocol):
    """Protocol implemented by image embedding backends."""

    def embed_images(self, images: Iterable[Image.Image]) -> List[List[float]]:
        """Return embeddings for each ``Image`` in ``images``."""


@dataclass
class CLIPEmbeddingBackend:
    """Embedding backend that uses the open-source CLIP model."""

    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if any(obj is None for obj in (torch, CLIPModel, CLIPProcessor)):
            raise ImportError(
                "transformers[torch] is required for CLIPEmbeddingBackend"
            )
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    def embed_images(self, images: Iterable[Image.Image]) -> List[List[float]]:
        batched = self._processor(
            images=list(images),
            return_tensors="pt",
        )
        batched = {k: v.to(self.device) for k, v in batched.items()}
        with torch.no_grad():
            outputs = self._model.get_image_features(**batched)
        return outputs.cpu().tolist()


@dataclass
class OpenAIVisionEmbeddingBackend:
    """Embedding backend that uses OpenAI's GPT-4o vision embeddings."""

    api_key: str
    model_name: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        if OpenAI is None:  # pragma: no cover - runtime guard
            raise ImportError("openai python package is required for OpenAIVisionEmbeddingBackend")
        self._client = OpenAI(api_key=self.api_key)

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def embed_images(self, images: Iterable[Image.Image]) -> List[List[float]]:
        payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract a visual embedding for outfit similarity search.",
                    },
                    {
                        "type": "input_image",
                        "image_base64": self._encode_image(image),
                    },
                ],
            }
            for image in images
        ]
        embeddings = []
        for message in payload:
            response = self._client.responses.create(
                model=self.model_name,
                input=[message],
            )
            embeddings.append(response.output[0].content[0].embedding)  # type: ignore[index]
        return embeddings


@dataclass
class ImageEmbedder:
    """High level wrapper for embedding images."""

    backend: ImageEmbeddingBackend

    def encode(self, images: Iterable[Image.Image]) -> List[List[float]]:
        return self.backend.embed_images(images)
