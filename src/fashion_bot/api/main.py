"""FastAPI application exposing the AI fashion services."""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from functools import lru_cache
from typing import Dict

from fastapi import Depends, FastAPI, File, UploadFile
from PIL import Image

from ..embeddings.image import CLIPEmbeddingBackend, ImageEmbedder
from ..embeddings.text import OpenAIEmbeddingBackend, TextEmbedder
from ..services.chat import ChatStylist
from ..services.wardrobe import WardrobeRepository
from ..services.visualization import OutfitVisualizer, PromptEngineer
from ..vectorstores.faiss_store import FaissVectorStore

app = FastAPI(title="Fashion Bot Platform")


# Dependency factories ----------------------------------------------------

def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    return api_key


def vector_store_factory() -> FaissVectorStore:
    return FaissVectorStore(dimension=512)


wardrobe_repository = WardrobeRepository()


@lru_cache(maxsize=1)
def get_image_embedder() -> ImageEmbedder:
    return ImageEmbedder(CLIPEmbeddingBackend())


@app.post("/wardrobe/{user_id}")
async def create_wardrobe_item(
    user_id: str,
    category: str,
    description: str,
    image: UploadFile = File(...),
    api_key: str = Depends(get_openai_api_key),
):
    image_bytes = await image.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_embedder = get_image_embedder()
    text_embedder = TextEmbedder(OpenAIEmbeddingBackend(api_key=api_key))
    wardrobe = wardrobe_repository.get_or_create(
        user_id=user_id,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        vector_store_factory=vector_store_factory,
    )
    item = wardrobe.add_item(category=category, description=description, image=pil_image)
    return {"item_id": item.item_id, "category": item.category, "description": item.description}


@app.post("/wardrobe/{user_id}/suggestions")
async def get_outfit_suggestions(
    user_id: str,
    image: UploadFile = File(...),
    api_key: str = Depends(get_openai_api_key),
):
    wardrobe = wardrobe_repository.wardrobes.get(user_id)
    if not wardrobe:
        return {"suggestions": []}
    image_bytes = await image.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    suggestions = wardrobe.suggest_pairings(pil_image)
    return {
        "suggestions": [
            {"item_id": item.item_id, "category": item.category, "description": item.description}
            for item in suggestions
        ]
    }


@app.post("/chat/{user_id}")
async def chat_with_stylist(user_id: str, payload: Dict[str, str], api_key: str = Depends(get_openai_api_key)):
    wardrobe = wardrobe_repository.wardrobes.get(user_id)
    if not wardrobe:
        raise RuntimeError("Wardrobe not found")
    stylist = ChatStylist(
        api_key=api_key,
        wardrobe_store=wardrobe.vector_store,
        text_embedder=wardrobe.text_embedder,
    )
    reply = stylist.chat(payload["message"])
    return {"reply": reply}


@app.post("/visualise/{user_id}")
async def visualise_outfit(user_id: str, style_brief: Dict[str, str], api_key: str = Depends(get_openai_api_key)):
    prompt_engineer = PromptEngineer(api_key=api_key)
    prompt = prompt_engineer.build_prompt(style_brief)
    visualizer = OutfitVisualizer()
    image = visualizer.generate(prompt)
    output_path = Path("generated") / f"{user_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {"prompt": prompt, "image_path": str(output_path)}
