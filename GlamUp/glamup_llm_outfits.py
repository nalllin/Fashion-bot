import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
from pathlib import Path

load_dotenv()
client = OpenAI()

BASE_DIR = Path(r"C:\Users\Janaki\Fashion-bot\GlamUp")

ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(ENV_PATH)

SYSTEM_PROMPT = """
You are GlamUp, an AI fashion stylist.

Your job:
- Take a structured user profile and generate complete outfit ideas.
- Focus on modern, wearable, India-friendly casual looks.
- Respect the user's preferred styles, occasion and color palette.
- Use neutral, earthy palettes when requested (beige, tan, brown, olive, cream, white, black, muted tones).

You MUST return a JSON object with this structure:

{
  "outfits": [
    {
      "id": "outfit_1",
      "vibe": "short human-readable description",
      "items": {
        "top": {
          "name": "...",
          "color": "...",
          "style_tags": ["minimal", "elegant"],
          "notes": "any styling notes"
        },
        "bottom": {
          "name": "...",
          "color": "...",
          "style_tags": ["..."],
          "notes": "..."
        },
        "dress": null OR same structure as top,
        "shoes": {
          "name": "...",
          "color": "...",
          "style_tags": ["..."],
          "notes": "..."
        },
        "bag": {
          "name": "...",
          "color": "...",
          "style_tags": ["..."],
          "notes": "..."
        },
        "accessories": [
          {
            "name": "...",
            "color": "...",
            "style_tags": ["..."],
            "notes": "..."
          }
        ]
      }
    }
  ]
}

Rules:
- Each outfit either uses (top + bottom) OR a single dress, not both.
- Always include shoes.
- Try to include at least one accessory (earrings, bracelet, watch, ring, necklace).
- Make everything coherent with the user's profile.
- Keep names realistic, like product titles on a fashion site.
- Keep notes short and practical.
Return ONLY valid JSON, no extra text.
"""


def generate_outfits_from_profile(
    profile: Dict[str, Any],
    num_outfits: int = 3,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """
    Given a structured profile dict, call the LLM and return a list of outfit dicts.

    Example profile:
    {
        "preferred_styles": ["minimal", "elegant"],
        "occasion": "casual",
        "palette": ["neutral", "earthy"],
        "fit": ["relaxed"],
        "climate": "warm",
        "notes": "no heels, prefer flats"
    }
    """
    user_payload = {
        "profile": profile,
        "num_outfits": num_outfits,
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        response_format={"type": "json_object"},
        temperature=0.9,
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    # safety: ensure "outfits" exists and is a list
    outfits = data.get("outfits", [])
    if not isinstance(outfits, list):
        raise ValueError("Model response did not contain 'outfits' list")
    return outfits


if __name__ == "__main__":
    # quick manual test
    test_profile = {
        "preferred_styles": ["minimal", "elegant"],
        "occasion": "casual",
        "palette": ["neutral", "earthy"],
        "fit": ["relaxed"],
        "climate": "warm",
        "notes": "prefer flats over heels"
    }

    outfits = generate_outfits_from_profile(test_profile, num_outfits=3)
    print(json.dumps(outfits, indent=2, ensure_ascii=False))
