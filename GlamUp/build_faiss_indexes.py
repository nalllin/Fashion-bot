import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


BASE_DIR = Path(r"C:\Users\Janaki\Fashion-bot\GlamUp")
DATA_PATH = BASE_DIR / "data" / "products.csv"
EMB_DIR = BASE_DIR / "embeddings"
EMB_DIR.mkdir(exist_ok=True, parents=True)

EMB_MODEL = "text-embedding-3-small"
CATEGORIES = ["top", "bottom", "dress", "shoes", "bag", "accessories"]


def get_embeddings(texts, batch_size=64):
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model=EMB_MODEL,
            input=batch,
        )
        embs = np.array([d.embedding for d in resp.data], dtype="float32")
        all_embs.append(embs)

    embs = np.vstack(all_embs).astype("float32")
    # normalize for cosine similarity with IndexFlatIP
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    embs /= norms
    return embs


def build_index_for_category(df_cat, category):
    if df_cat.empty:
        print(f"[WARN] No products for category: {category}")
        return

    print(f"[INFO] Building index for {category} ({len(df_cat)} items)")

    # Text we embed for each product
    texts = []
    metadata = []

    for _, row in df_cat.iterrows():
        desc = (
            f"Name: {row['name']}. "
            f"Category: {row['category']}. "
            f"Description: {row.get('description', '')}."
        )
        texts.append(desc)
        metadata.append(
            {
                "id": str(row["id"]),
                "name": row["name"],
                "category": row["category"],
                "image_path": row["image_path"],
            }
        )

    embs = get_embeddings(texts)
    dim = embs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    idx_path = EMB_DIR / f"faiss_{category}.index"
    meta_path = EMB_DIR / f"metadata_{category}.json"

    faiss.write_index(index, str(idx_path))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved index -> {idx_path}")
    print(f"[OK] Saved metadata -> {meta_path}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"products.csv not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    for cat in CATEGORIES:
        df_cat = df[df["category"].str.lower() == cat]
        build_index_for_category(df_cat, cat)


if __name__ == "__main__":
    main()
