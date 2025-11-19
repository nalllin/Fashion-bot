# glamup_ui.py

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from glamup_llm_outfits import generate_outfits_from_profile
from glamup_matcher import match_outfits_to_catalog


# ---------- Load ROOT .env ----------
BASE_DIR = Path(r"C:\Users\Janaki\Fashion-bot\GlamUp")
ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)


# ---------- Helpers ----------
def resolve_image_path(rel_path: str) -> str:
    """Convert catalog image_path (relative) to an absolute path usable by Streamlit."""
    return str(BASE_DIR / rel_path)


def safe_open_image(path: str) -> Image.Image | None:
    if os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None


def build_outfit_collage(cat_outfit: dict, idx: int) -> str | None:
    """
    Create a 2x3 collage image for one outfit and save it under data/outfit_collages.
    Layout:
      [ Top / Dress ] [ Bottom or empty ] [ Shoes ]
      [ Bag          ] [ Accessories     ] [ empty ]
    Returns the saved collage path (absolute) or None if nothing could be built.
    """
    # Collect images
    top_or_dress = cat_outfit.get("dress") or cat_outfit.get("top")
    bottom = cat_outfit.get("bottom")
    shoes = cat_outfit.get("shoes")
    bag = cat_outfit.get("bag")
    accessories = cat_outfit.get("accessories")

    slots = [top_or_dress, bottom, shoes, bag, accessories]

    slot_images: list[Image.Image | None] = []
    for item in slots:
        if item:
            img = safe_open_image(resolve_image_path(item["image_path"]))
        else:
            img = None
        slot_images.append(img)

    if not any(slot_images):
        return None

    # Collage config
    cols, rows = 3, 2
    width, height = 360, 360      # size per tile
    gap = 8                       # padding between tiles

    # create blank canvas
    collage_w = cols * width + (cols + 1) * gap
    collage_h = rows * height + (rows + 1) * gap
    collage = Image.new("RGB", (collage_w, collage_h), color=(15, 15, 20))

    # helper to paste a tile at grid coordinates
    def paste_tile(img: Image.Image | None, c: int, r: int):
        x = gap + c * (width + gap)
        y = gap + r * (height + gap)
        if img is None:
            # draw empty tile rectangle
            empty_tile = Image.new("RGB", (width, height), color=(30, 30, 40))
            collage.paste(empty_tile, (x, y))
        else:
            # fit image into tile preserving aspect
            img_copy = img.copy()
            img_copy.thumbnail((width, height))
            # center it
            tile = Image.new("RGB", (width, height), color=(30, 30, 40))
            tx = (width - img_copy.width) // 2
            ty = (height - img_copy.height) // 2
            tile.paste(img_copy, (tx, ty))
            collage.paste(tile, (x, y))

    # place the 5 slots
    # row 0: [ top/dress, bottom, shoes ]
    paste_tile(slot_images[0], 0, 0)  # top_or_dress
    paste_tile(slot_images[1], 1, 0)  # bottom
    paste_tile(slot_images[2], 2, 0)  # shoes
    # row 1: [ bag, accessories, empty ]
    paste_tile(slot_images[3], 0, 1)  # bag
    paste_tile(slot_images[4], 1, 1)  # accessories

    # save
    out_dir = BASE_DIR / "data" / "outfit_collages"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"outfit_{idx}.png"
    collage.save(out_path)
    return str(out_path)


def build_profile_from_quiz(
    splurge_on,
    risk_level,
    body_type,
    disliked_colors,
    outfit_types,
    style_words,
    age,
):
    """Map quiz answers -> style profile dict."""
    preferred_styles = []

    if "Preppy" in style_words:
        preferred_styles.append("classic")
    if "Feminine" in style_words:
        preferred_styles.append("elegant")
    if "Professional" in style_words:
        preferred_styles.append("classic")
    if "Boho" in style_words:
        preferred_styles.append("boho")

    if risk_level in ("I like to experiment", "I'm up for anything!"):
        preferred_styles.append("edgy")
    if age is not None and age <= 27 and any(
        o in outfit_types for o in ["Night Out", "Date Night"]
    ):
        preferred_styles.append("streetwear")

    if not preferred_styles:
        preferred_styles = ["minimal", "elegant"]
    preferred_styles = list(dict.fromkeys(preferred_styles))

    if "Work" in outfit_types:
        occasion = "work"
    elif "Date Night" in outfit_types:
        occasion = "date"
    elif "Night Out" in outfit_types:
        occasion = "party"
    elif "Elevated Casual" in outfit_types:
        occasion = "casual"
    else:
        occasion = "casual"

    palette = []
    if any(c in disliked_colors for c in ["Black", "Navy", "Bright Blue"]):
        palette.extend(["neutral", "earthy", "pastel"])
    else:
        palette.extend(["neutral", "earthy"])
    palette = list(dict.fromkeys(palette))

    fit = []
    if body_type in ["Athletic", "Rectangle"]:
        fit.extend(["relaxed", "tailored"])
    elif body_type in ["Hourglass", "Pear"]:
        fit.extend(["fitted", "tailored"])
    elif body_type == "Petite":
        fit.append("fitted")
    elif body_type == "Apple":
        fit.append("relaxed")
    if risk_level in ("I like to experiment", "I'm up for anything!"):
        fit.append("oversized")
    if not fit:
        fit = ["relaxed"]
    fit = list(dict.fromkeys(fit))

    climate = "warm"

    notes_bits = []
    if splurge_on:
        notes_bits.append(
            "likes to splurge on " + ", ".join(s.lower() for s in splurge_on)
        )
    if disliked_colors:
        notes_bits.append(
            "does not like these colors: " + ", ".join(disliked_colors).lower()
        )
    if style_words:
        notes_bits.append("style keywords: " + ", ".join(style_words).lower())

    notes = (
        "; ".join(notes_bits)
        if notes_bits
        else "prefer flats over heels; nothing too flashy"
    )

    return {
        "preferred_styles": preferred_styles,
        "occasion": occasion,
        "palette": palette,
        "fit": fit,
        "climate": climate,
        "notes": notes,
    }


# =========================================================
#                       UI APP
# =========================================================

def main():
    st.set_page_config(page_title="GlamUp AI", layout="wide")
    st.title("✨ GlamUp — AI Outfit Cards from Your Catalog")

    st.markdown(
        "Answer a quick style quiz, let me fill your profile, "
        "then I’ll design outfits and match them to your catalog."
    )

    # =======================================================
    #              Mada-style Onboarding Quiz
    # =======================================================
    st.header("🧪 Quick Style Quiz")

    splurge_on = st.multiselect(
        "What are you willing to splurge on? (choose all that apply)",
        ["Tops", "Bottoms", "Dresses", "Shoes"],
        key="q_splurge_on",
    )

    risk_level = st.radio(
        "How comfortable are you with taking fashion risks?",
        [
            "I play it safe",
            "Once in a while I mix it up",
            "Depends on the day",
            "I like to experiment",
            "I'm up for anything!",
        ],
        key="q_risk_level",
    )

    body_type = st.radio(
        "What is your body type?",
        ["Petite", "Pear", "Apple", "Athletic", "Rectangle", "Hourglass"],
        key="q_body_type",
    )

    st.markdown("**What size do you typically wear?**")
    col_sz1, col_sz2 = st.columns(2)
    with col_sz1:
        dress_size = st.text_input("Dress Size", key="q_dress_size")
        pants_size = st.text_input("Pants Size", key="q_pants_size")
    with col_sz2:
        shoe_size = st.text_input("Shoe Size", key="q_shoe_size")
        shirt_size = st.text_input("Shirt Size", key="q_shirt_size")

    disliked_colors = st.multiselect(
        "What colors do you NOT like to wear?",
        ["White", "Black", "Navy", "Brown", "Beige", "Bright Blue"],
        key="q_disliked_colors",
    )

    outfit_types = st.multiselect(
        "What types of outfits are you looking for?",
        ["Elevated Casual", "Work", "Night Out", "Date Night"],
        key="q_outfit_types",
    )

    style_words = st.multiselect(
        "How would you describe your style?",
        ["Preppy", "Feminine", "Professional", "Boho"],
        key="q_style_words",
    )

    age = st.slider("How old are you?", min_value=16, max_value=80, value=24, key="q_age")

    if st.button("✨ Use quiz answers to fill my style profile"):
        derived_profile = build_profile_from_quiz(
            splurge_on,
            risk_level,
            body_type,
            disliked_colors,
            outfit_types,
            style_words,
            age,
        )
        for k, v in derived_profile.items():
            st.session_state[k] = v
        st.success("Updated your style profile from quiz answers!")

    st.markdown("---")

    # =======================================================
    #                     Profile Sidebar
    # =======================================================
    st.sidebar.header("Your Style Profile")

    if "preferred_styles" not in st.session_state:
        st.session_state.preferred_styles = ["minimal", "elegant"]
    if "occasion" not in st.session_state:
        st.session_state.occasion = "casual"
    if "palette" not in st.session_state:
        st.session_state.palette = ["neutral", "earthy"]
    if "fit" not in st.session_state:
        st.session_state.fit = ["relaxed"]
    if "climate" not in st.session_state:
        st.session_state.climate = "warm"
    if "notes" not in st.session_state:
        st.session_state.notes = "prefer flats over heels; nothing too flashy"

    preferred_styles = st.sidebar.multiselect(
        "Preferred styles",
        ["minimal", "elegant", "streetwear", "boho", "classic", "sporty", "edgy"],
        default=st.session_state.preferred_styles,
        key="preferred_styles",
    )

    occasion = st.sidebar.selectbox(
        "Occasion",
        ["casual", "work", "party", "festive", "date", "travel"],
        index=["casual", "work", "party", "festive", "date", "travel"].index(
            st.session_state.occasion
        ),
        key="occasion",
    )

    palette = st.sidebar.multiselect(
        "Palette",
        ["neutral", "earthy", "pastel", "bright", "monochrome", "jewel tones"],
        default=st.session_state.palette,
        key="palette",
    )

    fit = st.sidebar.multiselect(
        "Preferred fit",
        ["relaxed", "tailored", "oversized", "fitted"],
        default=st.session_state.fit,
        key="fit",
    )

    climate = st.sidebar.selectbox(
        "Climate",
        ["warm", "humid", "mild", "cool"],
        index=["warm", "humid", "mild", "cool"].index(st.session_state.climate),
        key="climate",
    )

    notes = st.sidebar.text_area(
        "Extra notes (optional)",
        value=st.session_state.notes,
        height=80,
        key="notes",
    )

    num_outfits = st.sidebar.slider(
        "Number of outfits", min_value=1, max_value=6, value=3, key="num_outfits"
    )

    profile = {
        "preferred_styles": preferred_styles,
        "occasion": occasion,
        "palette": palette,
        "fit": fit,
        "climate": climate,
        "notes": notes,
    }

    st.write("#### Current profile")
    st.json(profile, expanded=False)

    # =======================================================
    #                Generate Outfits
    # =======================================================
    if st.button("Generate My GlamUp Cards"):
        with st.spinner("Designing outfits with LLM..."):
            llm_outfits = generate_outfits_from_profile(
                profile=profile,
                num_outfits=num_outfits,
            )

        with st.spinner("Matching outfits to your catalog..."):
            catalog_outfits = match_outfits_to_catalog(llm_outfits)

        st.success(f"✨ Generated **{len(catalog_outfits)}** catalog outfits")

        for idx, (llm_outfit, cat_outfit) in enumerate(
            zip(llm_outfits, catalog_outfits), start=1
        ):
            st.markdown("")
            with st.container(border=True):
                st.subheader(f"Outfit {idx}: {llm_outfit.get('vibe', '')}")

                with st.expander("LLM outfit description", expanded=False):
                    st.json(llm_outfit.get("items", {}), expanded=False)

                collage_path = build_outfit_collage(cat_outfit, idx)
                if collage_path:
                    st.image(collage_path, use_container_width=True)
                else:
                    st.warning("Could not create collage for this outfit.")

    st.markdown("---")
    st.markdown(
        "💡 Tip: tweak **style, palette, and occasion** in the sidebar after using the quiz, "
        "then regenerate to see how outfits change."
    )


if __name__ == "__main__":
    main()
