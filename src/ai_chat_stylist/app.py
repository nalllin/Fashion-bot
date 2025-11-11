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

    for image_path in args.image:
        try:
            description = stylist.add_outfit_image(str(image_path))
            print(f"Loaded image '{image_path}': {description}\n")
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")

    print("Type 'exit' or 'quit' to end the session.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print()
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            response = stylist.chat(user_input)
        except ValueError as exc:
            print(f"Error: {exc}")
            continue
        print(f"Stylist: {response}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
