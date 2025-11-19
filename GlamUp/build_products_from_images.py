import csv
import uuid
from pathlib import Path

# Root of your GlamUp project
BASE_DIR = Path(r"C:\Users\Janaki\Fashion-bot\GlamUp")

IMAGES_ROOT = BASE_DIR / "data" / "images"
OUT_CSV = BASE_DIR / "data" / "products.csv"

# Map folder-name prefixes to normalized category labels
PREFIX_TO_CATEGORY = {
    "tops": "top",
    "bottoms": "bottom",
    "shoes": "shoes",
    "bags": "bag",
    "accessories": "accessories",
    "dresses": "dress",
}


def infer_category_from_relpath(rel_path: Path) -> str | None:
    """
    rel_path is relative to IMAGES_ROOT, e.g.
    accessories-20251116T220748Z-1-001/some_image.jpg

    We take the first part of the path (the top folder) and
    map it using PREFIX_TO_CATEGORY.
    """
    if not rel_path.parts:
        return None
    folder_name = rel_path.parts[0].lower()

    for prefix, cat in PREFIX_TO_CATEGORY.items():
        if folder_name.startswith(prefix):
            return cat
    return None


def main():
    if not IMAGES_ROOT.exists():
        raise FileNotFoundError(f"Images root not found: {IMAGES_ROOT}")

    rows = []
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}

    for img_path in IMAGES_ROOT.rglob("*"):
        if not img_path.is_file():
            continue

        if img_path.suffix.lower() not in img_exts:
            continue

        rel = img_path.relative_to(IMAGES_ROOT)
        category = infer_category_from_relpath(rel)
        if category is None:
            print(f"[SKIP] {rel} (could not infer category)")
            continue

        # Simple display name from filename
        name = img_path.stem.replace("_", " ").replace("-", " ").title()

        rows.append(
            {
                "id": str(uuid.uuid4()),
                "name": name,
                "category": category,
                # image_path stored relative to BASE_DIR so UI can resolve it
                "image_path": str(Path("data") / "images" / rel),
                "description": f"{category} piece called {name}",
            }
        )

    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "name", "category", "image_path", "description"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} products to {OUT_CSV}")


if __name__ == "__main__":
    main()
