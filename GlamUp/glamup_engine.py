import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()
client = OpenAI()

BASE_DIR = Path(r"C:\Users\Janaki\Fashion-bot\GlamUp")

ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(ENV_PATH)

EMB_DIR = BASE_DIR / "embeddings"

EMB_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Categories used by the UI
CATEGORIES = ["top", "bottom", "dress", "shoes", "bag", "accessories"]


# ---------- helpers ----------

def _load_category_index(category: str):
    idx_path = EMB_DIR / f"faiss_{category}.index"
    meta_path = EMB_DIR / f"metadata_{category}.json"

    if not idx_path.exists() or not meta_path.exists():
        return None, None

    index = faiss.read_index(str(idx_path))
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def _embed_text(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=[text],
    )
    emb = np.array(resp.data[0].embedding, dtype="float32")
    emb /= np.linalg.norm(emb) + 1e-10
    return emb


def _expand_profile(profile_text: str) -> Dict:
    """
    Use gpt-4o-mini to convert free text into a light structured profile.
    """
    system = (
        "You are an AI fashion stylist. Extract a clean JSON profile from the user text "
        "with keys: preferred_styles (list), occasion (string), palette (list), "
        "fit (list), notes (string). Return ONLY JSON."
    )

    payload = {
        "raw_profile": profile_text,
    }

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(resp.choices[0].message.content)


def _build_query_from_profile(profile: Dict) -> str:
    """
    Turn structured profile into a single text string to embed.
    """
    styles = ", ".join(profile.get("preferred_styles", []))
    palette = ", ".join(profile.get("palette", []))
    fit = ", ".join(profile.get("fit", []))
    occ = profile.get("occasion", "")
    notes = profile.get("notes", "")

    return (
        f"User prefers {styles} style outfits for {occ} occasions, "
        f"with {palette} colors, {fit} fits. Extra notes: {notes}"
    )


def _search_category(
    category: str,
    query_emb: np.ndarray,
    k: int = 10,
) -> List[Dict]:
    index, metadata = _load_category_index(category)
    if index is None or not metadata:
        return []

    if query_emb.ndim == 1:
        query_emb = query_emb[None, :]

    scores, idxs = index.search(query_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        item = metadata[int(idx)].copy()
        item["score"] = float(score)
        results.append(item)
    return results


# ---------- public API used by Streamlit ----------

def get_glamup_outfits(
    profile_text: str,
    num_outfits: int = 3,
    per_category_pool: int = 5,
) -> List[Dict]:
    """
    Main entry point for the UI.

    profile_text: free text description of style / occasion / vibe
    num_outfits: how many outfits to generate
    returns: list of dicts, each with keys matching CATEGORIES,
             each value a product dict with at least name + image_path.
    """
    # 1) expand to structured profile
    profile = _expand_profile(profile_text)

    # 2) build a single query string & embed it
    query_text = _build_query_from_profile(profile)
    query_emb = _embed_text(query_text)

    # 3) get pools of items per category from FAISS
    pools: Dict[str, List[Dict]] = {}
    for cat in CATEGORIES:
        res = _search_category(cat, query_emb, k=per_category_pool)
        # if no items for a category (e.g. dress missing), just leave empty list
        pools[cat] = res

    # 4) assemble outfits (simple round-robin / best available)
    outfits: List[Dict] = []
    for i in range(num_outfits):
        outfit = {}
        # if we have dresses, alternate between dress-only and top+bottom
        use_dress = bool(pools.get("dress")) and (i % 2 == 1)

        if use_dress:
            # Pick dress
            dress_pool = pools.get("dress", [])
            if dress_pool:
                outfit["dress"] = dress_pool[i % len(dress_pool)]
            # No separate top/bottom in this case
            outfit["top"] = None
            outfit["bottom"] = None
        else:
            # Top
            top_pool = pools.get("top", [])
            outfit["top"] = top_pool[i % len(top_pool)] if top_pool else None

            # Bottom
            bottom_pool = pools.get("bottom", [])
            outfit["bottom"] = (
                bottom_pool[i % len(bottom_pool)] if bottom_pool else None
            )

            # No dress here
            outfit["dress"] = None

        # Shoes
        shoes_pool = pools.get("shoes", [])
        outfit["shoes"] = (
            shoes_pool[i % len(shoes_pool)] if shoes_pool else None
        )

        # Bag
        bag_pool = pools.get("bag", [])
        outfit["bag"] = bag_pool[i % len(bag_pool)] if bag_pool else None

        # Accessories
        acc_pool = pools.get("accessories", [])
        outfit["accessories"] = (
            acc_pool[i % len(acc_pool)] if acc_pool else None
        )

        outfits.append(outfit)

    return outfits
