"""Outfit visualisation helpers leveraging Stable Diffusion XL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    import torch
    from diffusers import StableDiffusionXLPipeline
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    StableDiffusionXLPipeline = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


@dataclass
class OutfitVisualizer:
    """Generate outfit concepts with Stable Diffusion XL."""

    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if StableDiffusionXLPipeline is None:
            raise ImportError("diffusers[torch] is required for OutfitVisualizer")
        resolved_device = self.device
        if torch is not None and self.device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"
        self._pipe = StableDiffusionXLPipeline.from_pretrained(self.model_path)
        self._pipe.to(resolved_device)
        self.device = resolved_device

    def generate(self, prompt: str, guidance_scale: float = 7.5, steps: int = 30):
        image = self._pipe(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=steps).images[0]
        return image


@dataclass
class PromptEngineer:
    """Uses GPT-4o-mini to craft prompts for outfit generation."""

    api_key: str
    model_name: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        if OpenAI is None:
            raise ImportError("openai python package is required for PromptEngineer")
        self._client = OpenAI(api_key=self.api_key)

    def build_prompt(self, style_brief: Dict[str, str]) -> str:
        message = (
            "Create a Stable Diffusion prompt for an outfit visualisation. "
            "Highlight garments, colours, and ambience based on the provided style brief.\n"
            f"Brief: {style_brief}"
        )
        response = self._client.responses.create(
            model=self.model_name,
            input=[{"role": "user", "content": message}],
        )
        return response.output[0].content[0].text  # type: ignore[index]
