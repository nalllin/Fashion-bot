"""
glamup.py

GlamUp outfit visualisation module for Alle (AI stylist).

Responsibilities:
- Connect to Postgres (DATABASE_URL env var).
- Store outfit items (image path/url, category, tags, embeddings).
- Ingest a local dataset of clothing images into the DB.
- Use OpenAI embeddings to rank items for a given style prompt.
- Build a simple horizontal outfit card image from selected items.

Dependencies (install these in your venv):
    pip install sqlalchemy psycopg2-binary pillow openai python-dotenv

You can later swap OpenAI embeddings for CLIP or your own model.
"""

from __future__ import annotations

import os
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from openai import OpenAI
import replicate

Base = declarative_base()


class OutfitItem(Base):
    """
    Simple outfit item table.

    Each row represents 1 product / clothing item in your catalog
    that you want to use for visual outfit cards.
    """

    __tablename__ = "glamup_outfit_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Optional: link back to your Medusa / marketplace product
    product_id = Column(String, nullable=True, index=True)
    sku = Column(String, nullable=True, index=True)

    # Where the image can be loaded from: local path or URL
    image_path = Column(String, nullable=False)

    # Basic metadata
    category = Column(String, nullable=False)  # e.g. "top", "bottom", "shoes"
    color = Column(String, nullable=True)
    style_tags = Column(String, nullable=True)  # comma-separated tags

    # Text embedding for semantic search
    embedding = Column(ARRAY(Float), nullable=True)



@dataclass
class GlamUpConfig:
    # Where to connect for outfits DB
    db_url: str = os.environ.get("DATABASE_URL", "")

    # Embeddings + LLM
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # How many items per outfit card (top, bottom, shoes, bag, accessories)
    max_items_per_outfit: int = 4

    # Image generation backend (we're using Replicate SDXL Lightning)
    image_backend: str = "replicate"
    replicate_model: str = (
        "bytedance/sdxl-lightning-4step:6f7a773af6fc3e8de9d5a3c00be77c17308914bf67772726aff83496ba1e3bbe"
    )


class GlamUp:
    def __init__(self, config: Optional[GlamUpConfig] = None) -> None:
        # Load config
        self.config = config or GlamUpConfig()

        # ✅ ALWAYS initialize OpenAI client first (for plan_outfit, embeddings, etc.)
        self.client = OpenAI()

        # DB attributes – may stay None if DB is not available
        self.engine = None
        self.SessionLocal = None

        # Get DB URL from config or env
        db_url = self.config.db_url or os.environ.get("DATABASE_URL", "")

        if not db_url:
            print("[GlamUp] No DATABASE_URL set – running in no-DB mode.")
            return

        # Normalize old-style postgres:// to SQLAlchemy's expected form
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)

        try:
            print("[GlamUp] Trying DB URL:", db_url)
            engine = create_engine(db_url)
            Base.metadata.create_all(engine)
            self.engine = engine
            self.SessionLocal = sessionmaker(bind=engine)
            print("[GlamUp] DB connected OK.")
        except Exception as exc:
            print("[GlamUp] WARNING: could not connect to DB, running in no-DB mode.")
            print("         Error:", exc)
            self.engine = None
            self.SessionLocal = None


    # ---------------------------------------------------------------------
    # Embeddings helpers
    # ---------------------------------------------------------------------
    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using OpenAI's embeddings API.
        """
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=texts,
        )
        return [d.embedding for d in response.data]

    # ---------------------------------------------------------------------
    # Ingestion
    # ---------------------------------------------------------------------
    def ingest_folder(
        self,
        root_dir: str,
        default_category: str,
        product_prefix: Optional[str] = None,
        dry_run: bool = False,
    ) -> int:
        """
        Ingest all image files in a folder into the DB.

        This is a simple helper for your local dataset where:
        - root_dir has only items of one category (e.g. "tops").
        - image filenames can optionally encode product/sku info.

        Arguments:
            root_dir: folder containing images.
            default_category: e.g. "top", "bottom", "shoes".
            product_prefix: optional prefix used for product_id (e.g. "prod_").
            dry_run: if True, only prints what would be inserted.

        Returns:
            Number of rows inserted.
        """
        import glob

        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            paths.extend(glob.glob(os.path.join(root_dir, ext)))

        if not paths:
            print(f"[GlamUp] No images found under: {root_dir}")
            return 0

        print(f"[GlamUp] Found {len(paths)} images in {root_dir}")

        # Make simple descriptions out of filenames for embeddings
        texts = []
        meta = []
        for p in paths:
            filename = os.path.basename(p)
            # Basic heuristic: use filename as description
            text_desc = filename.replace("_", " ").replace("-", " ")
            texts.append(f"{default_category} {text_desc}")
            # simple product_id based on filename if prefix provided
            prod_id = f"{product_prefix}{os.path.splitext(filename)[0]}" if product_prefix else None
            meta.append((p, prod_id))

        embeddings = self._embed_text(texts)

        if dry_run:
            print("[GlamUp] Dry run: would insert", len(paths), "items")
            return 0

        with self.SessionLocal() as session:
            for (img_path, prod_id), emb in zip(meta, embeddings):
                item = OutfitItem(
                    product_id=prod_id,
                    image_path=img_path,
                    category=default_category,
                    color=None,
                    style_tags=None,
                    embedding=emb,
                )
                session.add(item)
            session.commit()
        print(f"[GlamUp] Inserted {len(paths)} items into glamup_outfit_items.")
        return len(paths)

    # ---------------------------------------------------------------------
    # Outfit planning + retrieval
    # ---------------------------------------------------------------------
    def plan_outfit(self, user_prompt: str) -> List[str]:
        """
        Use LLM to infer which categories to include in the outfit.

        Returns a list of categories in order, e.g. ["top", "bottom", "shoes"].
        """
        system = (
            "You are a fashion planner. The user describes a look they want. "
            "Return ONLY a comma-separated list of clothing categories in order, "
            "using this vocabulary: top, bottom, dress, outerwear, shoes, bag, accessory."
        )
        resp = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=50,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip().lower()
        # Example output: "top, bottom, shoes"
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            return ["top", "bottom", "shoes"]
        # limit
        return parts[: self.config.max_items_per_outfit]

    def _similar_items_for_prompt(
        self,
        session: Session,
        prompt: str,
        category: str,
        top_k: int = 5,
    ) -> List[OutfitItem]:
        """
        Return items in a category sorted by cosine similarity to the prompt.
        """
        from math import sqrt

        emb = self._embed_text([f"{category} {prompt}"])[0]

        # Fetch all items for this category
        stmt = select(OutfitItem).where(OutfitItem.category == category)
        rows = session.execute(stmt).scalars().all()
        if not rows:
            return []

        def cosine(a: List[float], b: List[float]) -> float:
            if not a or not b:
                return -1.0
            dot = sum(x * y for x, y in zip(a, b))
            na = sqrt(sum(x * x for x in a))
            nb = sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return -1.0
            return dot / (na * nb)

        ranked = []
        for row in rows:
            if not row.embedding:
                continue
            sim = cosine(emb, row.embedding)
            ranked.append((sim, row))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in ranked[:top_k]]

    def recommend_outfit_items(
        self,
        user_prompt: str,
    ) -> List[OutfitItem]:
        categories = self.plan_outfit(user_prompt)
        print(f"[GlamUp] Planned categories: {categories}")

        # If no DB, we can't choose real catalog items – return empty list
        if self.SessionLocal is None:
            print("[GlamUp] No DB session – returning empty items list.")
            return []

        selected: List[OutfitItem] = []
        with self.SessionLocal() as session:
            for cat in categories:
                candidates = self._similar_items_for_prompt(
                    session=session,
                    prompt=user_prompt,
                    category=cat,
                    top_k=5,
                )
                if candidates:
                    selected.append(candidates[0])
        return selected


    # ---------------------------------------------------------------------
    # Card visualisation
    # ---------------------------------------------------------------------
    def build_outfit_card(
        self,
        items: List[OutfitItem],
        thumb_size: Tuple[int, int] = (512, 512),
        bg_color: Tuple[int, int, int] = (10, 10, 18),
    ) -> Image.Image:
        """
        Build a simple horizontal card that shows all item images side by side.

        Assumes image_path points to a local file. If you store URLs instead,
        you can swap this to download images first.
        """
        if not items:
            # create empty image
            return Image.new("RGB", (thumb_size[0], thumb_size[1]), bg_color)

        thumbs = []
        for item in items:
            try:
                img = Image.open(item.image_path).convert("RGB")
            except Exception as exc:
                print(f"[GlamUp] Failed to open image {item.image_path}: {exc}")
                continue
            img = img.resize(thumb_size)
            thumbs.append(img)

        if not thumbs:
            return Image.new("RGB", (thumb_size[0], thumb_size[1]), bg_color)

        width = thumb_size[0] * len(thumbs)
        height = thumb_size[1]
        card = Image.new("RGB", (width, height), bg_color)

        x = 0
        for t in thumbs:
            card.paste(t, (x, 0))
            x += thumb_size[0]

        return card

    # ---------------------------------------------------------------------
    # High-level one-shot API
    # ---------------------------------------------------------------------
    def glamup_card_from_prompt(
        self,
        user_prompt: str,
    ) -> Tuple[List[OutfitItem], Image.Image]:
        """
        One-call interface:

        1) Plan outfit categories with an LLM.
        2) Pick items for each category with embedding similarity.
        3) Build a horizontal outfit card.

        Returns (items, card_image).
        """
        items = self.recommend_outfit_items(user_prompt)
        card = self.build_outfit_card(items)
        return items, card
    
    # ---------------------------------------------------------------------
    # Outfit image generation
    # ---------------------------------------------------------------------
    def build_image_prompt_from_items(
        self,
        user_prompt: str,
        items: List[OutfitItem],
        extra_style: Optional[str] = None,
    ) -> str:
        """
        Turn selected items into a rich prompt for the image model.

        user_prompt: original user description, e.g.
            "soft girl college outfit for Mumbai humidity with white sneakers"

        items: the OutfitItem rows chosen by GlamUp.

        extra_style: optional extra art direction, e.g.
            "Pinterest-style flat lay, soft natural light, 3:4 vertical"
        """
        parts = []
        for item in items:
            base = item.category
            # Use filename as a crude descriptor; you can replace this with real product metadata
            filename = os.path.basename(item.image_path)
            name_hint = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
            desc = f"{base}: {name_hint}"
            if item.color:
                desc += f", color: {item.color}"
            if item.style_tags:
                desc += f", style: {item.style_tags}"
            parts.append(desc)

        outfit_desc = "; ".join(parts) if parts else "cohesive outfit with top, bottom, shoes, bag, and accessories"

        style_hint = extra_style or (
            "high quality fashion photograph, full outfit visible, "
            "clean studio background, soft natural lighting, pinterest aesthetic"
        )

        full_prompt = (
            f"Generate a single, cohesive outfit visualisation. "
            f"User request: {user_prompt}. "
            f"Outfit components: {outfit_desc}. "
            f"Style: {style_hint}."
        )
        return full_prompt
    
    def generate_outfit_image_from_items(
        self,
        user_prompt: str,
        items: List[OutfitItem],
        extra_style: Optional[str] = None,
    ) -> Optional[str]:

        img_prompt = self.build_image_prompt_from_items(
            user_prompt=user_prompt,
            items=items,
            extra_style=extra_style,
        )

        return self._generate_image_sdxl(
            prompt=img_prompt,
            width=1024,
            height=1024,
        )


    def glamup_card_with_generated_image(
        self,
        user_prompt: str,
        extra_style: Optional[str] = None,
    ) -> Tuple[List[OutfitItem], Image.Image, Optional[str]]:
        """
        1) Choose items from DB based on user_prompt.
        2) Build a horizontal catalog card from real product images.
        3) Generate a synthetic outfit image via SDXL.

        Returns:
            (items, card_image, generated_image_url)
        """
        items = self.recommend_outfit_items(user_prompt)
        card = self.build_outfit_card(items)

        gen_url = self.generate_outfit_image_from_items(
            user_prompt=user_prompt,
            items=items,
            extra_style=extra_style,
        )
        return items, card, gen_url

    def _generate_image_sdxl(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_outputs: int = 1,
    ) -> Optional[str]:
        """
        Generate an outfit image using SDXL-Lightning via Replicate.
        """
        if not os.environ.get("REPLICATE_API_TOKEN"):
            print("[GlamUp] Missing REPLICATE_API_TOKEN")
            return None

        try:
            output = replicate.run(
                self.config.replicate_model,
                input={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "guidance_scale": 0,       # recommended for lightning
                    "num_inference_steps": 4,  # this model is 4-step lightning
                    "negative_prompt": "low quality, distorted, ugly, bad anatomy"
                }
            )

            # Replicate returns a list of URLs
            if isinstance(output, list) and output:
                return output[0]

            print("[GlamUp] Unexpected SDXL output:", output)
            return None

        except Exception as exc:
            print("[GlamUp] SDXL generation error:", exc)
            return None






if __name__ == "__main__":
    # Simple test stub: requires env OPENAI_API_KEY and DATABASE_URL
    g = GlamUp()
    print("[GlamUp] Example: ingesting a 'tops' folder (dry run).")
    g.ingest_folder("./data/tops", default_category="top", dry_run=True)

    prompt = "soft girl college outfit for Mumbai humidity with white sneakers"
    items, card = g.glamup_card_from_prompt(prompt)
    print(f"[GlamUp] Selected {len(items)} items for prompt: {prompt}")

    # Save card to disk
    os.makedirs("glamup_cards", exist_ok=True)
    out_path = os.path.join("glamup_cards", "example_card.jpg")
    card.save(out_path)
    print("[GlamUp] Card saved to", out_path)
