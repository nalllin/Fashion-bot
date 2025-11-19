# glamup_matcher.py
#
# Maps LLM-generated outfits to closest items in the catalog using FAISS.

import json
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from pathlib import Path

load_dotenv()
client = OpenAI()

BASE_DIR = Path(r"C:\Users\Janaki\Fashion-bot\GlamUp")

ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(ENV_PATH)


# --- paths / config ---

BASE_DIR = Path(r"C:\Users\Janaki\Fashion-bot\GlamUp")
EMB_DIR = BASE_DIR / "embeddings"

EMB_MODEL = "text-embedding-3-small"

# Slots we care about (match your outfit structure)
CATEGORIES = ["top", "bottom", "dress", "shoes", "bag", "accessories"]


# ---------- helpers ----------

def _load_category_index(category: str):
    """
    Load FAISS index + metadata for a given category.
    Returns (index, metadata_list) or (None, None) if missing.
    """
    idx_path = EMB_DIR / f"faiss_{category}.index"
    meta_path = EMB_DIR / f"metadata_{category}.json"

    if not idx_path.exists() or not meta_path.exists():
        return None, None

    index = faiss.read_index(str(idx_path))
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def _embed_text(text: str) -> np.ndarray:
    """
    Get a normalized embedding for a text string.
    """
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=[text],
    )
    emb = np.array(resp.data[0].embedding, dtype="float32")
    emb /= np.linalg.norm(emb) + 1e-10
    return emb


def _search_best_match(
    category: str,
    query_text: str,
) -> Optional[Dict]:
    """
    For a given category and query text, return the best matching catalog item
    (or None if no index/items).
    """
    index, metadata = _load_category_index(category)
    if index is None or not metadata:
        return None

    q_emb = _embed_text(query_text)
    q_emb = q_emb[None, :]  # shape (1, dim)

    scores, idxs = index.search(q_emb, k=1)
    idx = idxs[0][0]
    if idx == -1:
        return None

    item = metadata[int(idx)].copy()
    item["score"] = float(scores[0][0])
    return item


def _build_query_for_item(slot: str, item: Dict, outfit_vibe: str = "") -> str:
    """
    Combine item fields into a single text to embed.
    Works with the structure from glamup_llm_outfits (name, color, style_tags, notes).
    """
    name = item.get("name", "")
    color = item.get("color", "")
    style_tags = ", ".join(item.get("style_tags", []))
    notes = item.get("notes", "")

    parts = [
        f"Slot: {slot}",
        f"Name: {name}",
        f"Color: {color}",
        f"Style: {style_tags}",
        f"Notes: {notes}",
        f"Outfit vibe: {outfit_vibe}",
    ]
    return ". ".join([p for p in parts if p])


# ---------- main API ----------

def match_outfits_to_catalog(llm_outfits: List[Dict]) -> List[Dict]:
    """
    Input: list of outfits from generate_outfits_from_profile()
           each looks like:
           {
             "id": "outfit_1",
             "vibe": "...",
             "items": {
               "top": {...},
               "bottom": {...},
               "dress": null or {...},
               "shoes": {...},
               "bag": {...},
               "accessories": [{...}, ...]
             }
           }

    Output: list of outfits with catalog items (for UI):
           {
             "vibe": "...",
             "top": {catalog_item or None},
             "bottom": {... or None},
             "dress": {... or None},
             "shoes": {... or None},
             "bag": {... or None},
             "accessories": {... or None}  # single best accessory
           }
    """
    mapped_outfits: List[Dict] = []

    for outfit in llm_outfits:
        vibe = outfit.get("vibe", "")
        o_items = outfit.get("items", {})

        mapped = {"vibe": vibe}

        # --- Top / Bottom / Dress ---

        # dress (if present)
        dress = o_items.get("dress")
        if dress:
            q = _build_query_for_item("dress", dress, vibe)
            mapped["dress"] = _search_best_match("dress", q)
            mapped["top"] = None
            mapped["bottom"] = None
        else:
            # top
            top = o_items.get("top")
            if top:
                q = _build_query_for_item("top", top, vibe)
                mapped["top"] = _search_best_match("top", q)
            else:
                mapped["top"] = None

            # bottom
            bottom = o_items.get("bottom")
            if bottom:
                q = _build_query_for_item("bottom", bottom, vibe)
                mapped["bottom"] = _search_best_match("bottom", q)
            else:
                mapped["bottom"] = None

            mapped["dress"] = None

        # shoes
        shoes = o_items.get("shoes")
        if shoes:
            q = _build_query_for_item("shoes", shoes, vibe)
            mapped["shoes"] = _search_best_match("shoes", q)
        else:
            mapped["shoes"] = None

        # bag
        bag = o_items.get("bag")
        if bag:
            q = _build_query_for_item("bag", bag, vibe)
            mapped["bag"] = _search_best_match("bag", q)
        else:
            mapped["bag"] = None

        # accessories (we just pick the first suggestion & match it)
        acc_list = o_items.get("accessories") or []
        if acc_list:
            acc = acc_list[0]
            q = _build_query_for_item("accessories", acc, vibe)
            mapped["accessories"] = _search_best_match("accessories", q)
        else:
            mapped["accessories"] = None

        mapped_outfits.append(mapped)

    return mapped_outfits
