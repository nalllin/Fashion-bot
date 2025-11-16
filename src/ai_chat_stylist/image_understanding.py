"""Utilities for converting outfit images into descriptive text."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from openai import OpenAI


def _load_image_bytes(image_path: str | Path) -> bytes:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path.read_bytes()


def describe_outfit_image(image_path: str | Path, prompt: Optional[str] = None) -> str:
    """Generate a textual summary for an outfit image using GPT-4o vision."""
    prompt = prompt or (
        "Provide a concise yet vivid description of this outfit, capturing key "
        "clothing items, their colors, fabric textures, and the overall vibe."
    )
    image_bytes = _load_image_bytes(image_path)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    client = OpenAI()
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_base64": b64_image},
                ],
            }
        ],
    )
    return response.output_text.strip()
