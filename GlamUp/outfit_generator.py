"""CLI wrapper for GlamUp outfit generation utilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from glamup_llm_outfits import generate_outfits_from_profile


def load_profile(path: Optional[Path]) -> Dict[str, Any]:
    """Load a style profile from JSON (path optional for quick demos)."""
    if path is None:
        # lightweight default profile so `python outfit_generator.py`
        # produces something meaningful out of the box.
        return {
            "preferred_styles": ["minimal", "elegant"],
            "occasion": "casual",
            "palette": ["neutral", "earthy"],
            "fit": ["relaxed"],
            "climate": "warm",
            "notes": "prefer flats over heels",
        }

    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Profile JSON must contain an object at the top level.")
    return data


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate GlamUp outfits using the LLM helper."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Path to a JSON file describing the user's style profile.",
    )
    parser.add_argument(
        "--num-outfits",
        type=int,
        default=3,
        help="Number of outfits to request from the LLM.",
    )

    args = parser.parse_args(argv)
    profile = load_profile(args.profile)
    outfits = generate_outfits_from_profile(
        profile=profile, num_outfits=args.num_outfits
    )
    print(json.dumps(outfits, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
