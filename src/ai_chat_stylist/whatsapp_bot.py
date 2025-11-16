"""WATI/WhatsApp webhook helpers for the AI stylist."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests

from .chatbot import AIChatStylist


@dataclass
class WatiIncomingMessage:
    """Light-weight representation of a message delivered by WATI."""

    wa_id: str
    body: str
    message_id: str
    message_type: str = "text"


def _append_message(
    collector: list[WatiIncomingMessage],
    *,
    wa_id: Optional[str],
    body: Optional[str],
    message_id: Optional[str],
    message_type: str,
) -> None:
    if wa_id and body:
        collector.append(
            WatiIncomingMessage(
                wa_id=wa_id,
                body=body,
                message_id=message_id or "",
                message_type=message_type,
            )
        )


def extract_wati_messages(payload: dict) -> list[WatiIncomingMessage]:
    """Parse a WATI webhook payload into normalized message objects."""
    messages: list[WatiIncomingMessage] = []

    # 1) Direct WATI webhook shape
    for message in payload.get("messages", []) or []:
        message_type = message.get("type", "text")
        wa_id = message.get("from") or message.get("waId") or payload.get("waId")
        message_id = message.get("id") or message.get("msgId")
        body: Optional[str] = None
        if message_type == "text":
            body = (message.get("text") or {}).get("body") or message.get("text")
        elif message_type == "interactive":
            interactive = message.get("interactive", {})
            if interactive.get("type") == "button_reply":
                body = (interactive.get("button_reply") or {}).get("title")
            elif interactive.get("type") == "list_reply":
                body = (interactive.get("list_reply") or {}).get("title")
        elif "message" in message:
            body = (message["message"].get("text") or {}).get("body")
        _append_message(
            messages,
            wa_id=wa_id,
            body=body,
            message_id=message_id,
            message_type=message_type,
        )

    # 2) Meta-style entries (forwarded straight from WhatsApp Cloud)
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {})
            for message in value.get("messages", []) or []:
                message_type = message.get("type", "text")
                wa_id = message.get("from")
                message_id = message.get("id")
                body: Optional[str] = None
                if message_type == "text":
                    body = (message.get("text") or {}).get("body")
                elif message_type == "interactive":
                    interactive = message.get("interactive", {})
                    if interactive.get("type") == "button_reply":
                        body = (interactive.get("button_reply") or {}).get("title")
                    elif interactive.get("type") == "list_reply":
                        body = (interactive.get("list_reply") or {}).get("title")
                _append_message(
                    messages,
                    wa_id=wa_id,
                    body=body,
                    message_id=message_id,
                    message_type=message_type,
                )

    # 3) Flat payload (single event)
    if not messages:
        wa_id = payload.get("waId") or payload.get("from")
        text_value: Optional[str] = None
        if isinstance(payload.get("text"), dict):
            text_value = payload["text"].get("body")
        elif isinstance(payload.get("text"), str):
            text_value = payload["text"]
        elif isinstance(payload.get("message"), dict):
            inner = payload["message"]
            if inner.get("type") == "text":
                text_value = (inner.get("text") or {}).get("body")
        if wa_id and text_value:
            messages.append(
                WatiIncomingMessage(
                    wa_id=wa_id,
                    body=text_value,
                    message_id=payload.get("msgId", ""),
                    message_type="text",
                )
            )

    return messages


def build_quick_reply_menu_payload() -> dict:
    """Return a WATI interactive button payload that mirrors the mock."""
    return {
        "header": {"type": "text", "text": "\ud83d\udc84 Your Fashion Assistant"},
        "body": {
            "text": (
                "What would you like to do?\n"
                "Choose an option 👇"
            )
        },
        "footer": {"text": "powered by AIChatStylist"},
        "buttons": [
            {
                "type": "reply",
                "title": "📤 Upload Wardrobe",
                "id": "upload-wardrobe",
            },
            {
                "type": "reply",
                "title": "📜 Ask Fashion Advice",
                "id": "ask-fashion-advice",
            },
            {
                "type": "reply",
                "title": "👚 Outfit Suggestion",
                "id": "outfit-suggestion",
            },
        ],
    }


class WatiTransport:
    """Minimal transport for the WATI API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not base_url:
            raise RuntimeError("WATI base URL is required")
        if not api_key:
            raise RuntimeError("WATI API key is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls) -> "WatiTransport":
        base_url = os.environ.get("WATI_SERVER_URL")
        api_key = os.environ.get("WATI_API_KEY")
        if not base_url or not api_key:
            raise RuntimeError(
                "WATI_SERVER_URL and WATI_API_KEY environment variables must be set."
            )
        return cls(base_url=base_url, api_key=api_key)

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._session.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"status": response.status_code}

    def send_text(self, to: str, body: str) -> dict:
        payload = {"messageText": body}
        return self._post(f"api/v1/sendSessionMessage/{to}", payload)

    def send_quick_reply_menu(self, to: str) -> dict:
        payload = build_quick_reply_menu_payload()
        return self._post(f"api/v1/sendInteractiveButtons/{to}", payload)


class WatiStylistBot:
    """Routes WATI webhook events to the AI stylist and sends replies."""

    def __init__(
        self,
        *,
        stylist: Optional[AIChatStylist] = None,
        transport: Optional[WatiTransport] = None,
        menu_triggers: Optional[Iterable[str]] = None,
    ) -> None:
        self.stylist = stylist or AIChatStylist()
        self.transport = transport
        triggers = menu_triggers or ("menu", "start", "hi", "hello", "help")
        self.menu_triggers = {trigger.lower() for trigger in triggers}

    def handle_payload(self, payload: dict) -> list[dict]:
        if self.transport is None:
            raise RuntimeError("WATI transport is not configured.")

        print("📩 RAW WATI PAYLOAD:", payload)

        responses: list[dict] = []
        messages = extract_wati_messages(payload)

        print("🧩 Parsed messages:", messages)

        for message in messages:
            print("📥 Incoming message.body:", repr(message.body))

            if not message.body:
                print("❌ message.body is EMPTY — cannot reply")
                continue

            normalized = message.body.strip().lower()
            if normalized in self.menu_triggers:
                print("🔘 Sending QUICK-REPLY menu")
                responses.append(self.transport.send_quick_reply_menu(message.wa_id))
                continue

            reply = self.stylist.chat(message.body)
            print("📤 LLM reply:", repr(reply))

            if not reply.strip():
                print("❌ LLM returned EMPTY reply")
                continue

            send_resp = self.transport.send_text(message.wa_id, reply)
            print("📨 WATI send response:", send_resp)

            responses.append(send_resp)

        return responses


def build_whati_webhook_response(payload: dict) -> List[dict]:
    """Convenience wrapper that wires env-configured WATI transport."""
    bot = WatiStylistBot(transport=WatiTransport.from_env())
    return bot.handle_payload(payload)
