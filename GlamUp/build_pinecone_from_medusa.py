# build_pinecone_from_medusa.py
#
# 1. Fetch products from Medusa /store/products
# 2. Build a rich text field per product
# 3. Upsert that text into a Pinecone index that has integrated
#    embedding with model=llama-text-embed-v2.

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from pinecone import Pinecone

from medusa_store import list_store_products
from pathlib import Path

load_dotenv()
#client = OpenAI()

BASE_DIR = Path(r"C:\Users\Janaki\Fashion-bot\GlamUp")

ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(ENV_PATH)


PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "slaylist-products")
NAMESPACE = "medusa-products"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)  # your existing integrated index


# ---------- Helpers to shape Medusa products ----------

def product_to_text(p: Dict[str, Any]) -> str:
    """Build a semantic description string that Pinecone will embed."""
    title = p.get("title", "")
    subtitle = p.get("subtitle") or ""
    description = p.get("description") or ""
    type_val = (p.get("type") or {}).get("value") or ""
    collection_title = (p.get("collection") or {}).get("title") or ""
    tags = " ".join(t.get("value", "") for t in p.get("tags", []))

    options_summary_parts = []
    for opt in p.get("options", []):
        opt_title = opt.get("title", "")
        opt_values = ", ".join(v.get("value", "") for v in opt.get("values", []))
        if opt_title or opt_values:
            options_summary_parts.append(f"{opt_title}: {opt_values}")
    options_summary = " | ".join(options_summary_parts)

    return (
        f"Product: {title}. "
        f"Subtitle: {subtitle}. "
        f"Description: {description}. "
        f"Type: {type_val}. "
        f"Collection: {collection_title}. "
        f"Tags: {tags}. "
        f"Options: {options_summary}."
    )


def safe_str(x):
    return "" if x is None else str(x)

def safe_list_str(lst):
    if not lst:
        return []
    return [safe_str(x) for x in lst]

def product_record(p: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Medusa product into a valid Pinecone record."""

    images = p.get("images", []) or []
    image_urls = []
    for img in images:
        url = img.get("url") or img.get("src") or img.get("thumbnail")
        if url:
            image_urls.append(url)

    thumb = p.get("thumbnail") or (image_urls[0] if image_urls else None)

    # Extract type, tags, collection safely
    type_val = (p.get("type") or {}).get("value")
    collection_val = (p.get("collection") or {}).get("title")
    tags = [t.get("value") for t in (p.get("tags") or [])]

    return {
        "_id": safe_str(p.get("id")),
        "text": safe_str(product_to_text(p)),   # required field for embedding

        # SAFE metadata fields
        "title": safe_str(p.get("title")),
        "subtitle": safe_str(p.get("subtitle")),
        "description": safe_str(p.get("description")),
        "collection": safe_str(collection_val),
        "type": safe_str(type_val),
        "tags": safe_list_str(tags),
        "thumbnail": safe_str(thumb),
        "images": safe_list_str(image_urls),
        "handle": safe_str(p.get("handle")),
    }




# ---------- Medusa paging ----------

def fetch_all_products(batch_size: int = 100) -> List[Dict[str, Any]]:
    all_products: List[Dict[str, Any]] = []
    offset = 0

    while True:
        data = list_store_products(limit=batch_size, offset=offset)
        products = data.get("products", [])
        if not products:
            break
        all_products.extend(products)
        offset += batch_size
        print(f"[INFO] fetched {len(products)} (total {len(all_products)})")
        if len(products) < batch_size:
            break

    return all_products


# ---------- Upsert into Pinecone (text only) ----------

def upsert_products_to_pinecone():
    products = fetch_all_products()
    print(f"[INFO] total products: {len(products)}")

    if not products:
        print("[WARN] no products from Medusa")
        return

        # Pinecone integrated indexes currently limit batch size to 96
    MAX_BATCH = 96

    for i in range(0, len(products), MAX_BATCH):
        batch = products[i : i + MAX_BATCH]
        records = [product_record(p) for p in batch]

        print(f"[INFO] upserting {len(records)} records to Pinecone...")
        index.upsert_records(
            namespace=NAMESPACE,
            records=records,
        )

    print("[OK] finished upserting all products ✅")


if __name__ == "__main__":
    upsert_products_to_pinecone()
