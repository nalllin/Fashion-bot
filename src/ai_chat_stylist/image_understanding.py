"""Utilities for converting outfit images into descriptive text."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from openai import OpenAI


def _load_image_bytes(image_path: str | Path) -> bytes:
    """Load image bytes from disk."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path.read_bytes()


def describe_outfit_image(image_path: str | Path, prompt: Optional[str] = None) -> str:
    """Generate a textual summary for an outfit image using GPT-4o vision.

    This uses the new OpenAI Python SDK (>=1.x) via the chat.completions API
    with an image_url payload.
    """
    prompt = prompt or (
        "Describe this outfit in a concise, fashion-aware way. "
        "Focus on: key garments, colors, fit/silhouette, and overall vibe."
    )

    image_bytes = _load_image_bytes(image_path)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_image}"

    client = OpenAI()  # uses OPENAI_API_KEY from the environment

    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o" if you prefer
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
    )

    # The text answer is in choices[0].message.content
    if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
        return resp.choices[0].message.content.strip()

    return "I couldn't interpret the outfit image clearly."
