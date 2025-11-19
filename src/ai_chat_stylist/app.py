"""Command line interface for the AI chat stylist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chatbot import AIChatStylist, StylistConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the AI stylist")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        help="Optional directory to persist FAISS index between runs.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="Path to an outfit image to pre-load into the conversation context.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override the default stylist personality.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = StylistConfig()
    if args.system_prompt:
        config.system_prompt = args.system_prompt

    stylist = AIChatStylist(config=config)

    # Preload outfit images into context
    for image_path in args.image:
        try:
            description = stylist.add_outfit_image(str(image_path))
            print(f"Loaded image '{image_path}': {description}\n")
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")

    print("------------------------------------------------------------")
    print(" Alle – Fashion AI Stylist (CLI Demo)")
    print("------------------------------------------------------------")
    print("Type your message normally to chat.")
    print("Type 'glamup: <your description>' to generate an outfit + SDXL image.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print()
            break

        if user_input.lower() in {"exit", "quit"}:
            break

        if not user_input:
            continue

        # ------------------------------------------
        # ✨ GLAMUP MODE
        # ------------------------------------------
        if user_input.lower().startswith("glamup:"):
            prompt = user_input.split(":", 1)[1].strip()

            print("\n✨ Generating GlamUp outfit, please wait...\n")
            try:
                summary, card_path, gen_url = stylist.glamup_outfit_with_image(prompt)

                print("------------------------------------------------------------")
                print("✨ GLAMUP RESULT")
                print("------------------------------------------------------------")
                print(summary)
                print(f"\n📌 Outfit card saved at: {card_path}")
                print(f"🖼️ SDXL image URL: {gen_url}")
                print("------------------------------------------------------------\n")

            except Exception as exc:
                print(f"Error running GlamUp: {exc}\n")

            continue

        # ------------------------------------------
        # 🤖 NORMAL CHAT
        # ------------------------------------------
        try:
            response = stylist.chat(user_input)
        except ValueError as exc:
            print(f"Error: {exc}")
            continue

        print(f"Alle: {response}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
