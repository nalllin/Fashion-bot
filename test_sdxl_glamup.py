"""Quick manual SDXL test that relies on environment variables instead of hard-coded secrets."""
from __future__ import annotations

import os
from pathlib import Path

from ai_chat_stylist.chatbot import AIChatStylist
from dotenv import load_dotenv


def ensure_env() -> None:
    """Load .env if present and fail loudly when required keys are missing."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)

    required = ["DATABASE_URL", "OPENAI_API_KEY", "REPLICATE_API_TOKEN"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Populate them in your shell or .env file before running the test."
        )


def main() -> None:
    ensure_env()
    stylist = AIChatStylist()  # this will create GlamUp() inside
    prompt = "soft girl college outfit with white sneakers for humid weather"
    summary, card_path, gen_url = stylist.glamup_outfit_with_image(prompt)

    print("SUMMARY:\n", summary)
    print("Card saved at:", card_path)
    print("SDXL image URL:", gen_url)


if __name__ == "__main__":
    main()
