"""Utilities for wiring the stylist into a WhatsApp/Whati bot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests

from .chatbot import AIChatStylist


@dataclass
class WhatsAppIncomingMessage:
    """Light-weight representation of a WhatsApp message payload."""

    wa_id: str
    body: str
    message_id: str
    message_type: str = "text"


def extract_whatsapp_messages(payload: dict) -> list[WhatsAppIncomingMessage]:
    """Parse WhatsApp/Whati webhook payload into normalized messages."""

    messages: list[WhatsAppIncomingMessage] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for message in value.get("messages", []) or []:
                wa_id = message.get("from")
                message_id = message.get("id", "")
                message_type = message.get("type", "text")
                body = ""
                if message_type == "text":
                    body = message.get("text", {}).get("body", "")
                elif message_type == "interactive":
                    interactive = message.get("interactive", {})
                    if interactive.get("type") == "button_reply":
                        body = interactive.get("button_reply", {}).get("title", "")
                    elif interactive.get("type") == "list_reply":
                        body = interactive.get("list_reply", {}).get("title", "")
                if wa_id and body:
                    messages.append(
                        WhatsAppIncomingMessage(
                            wa_id=wa_id,
                            body=body,
                            message_id=message_id,
                            message_type=message_type,
                        )
                    )
    return messages


def build_quick_reply_menu() -> dict:
    """Return an interactive message payload that mirrors the design mock."""

    return {
        "messaging_product": "whatsapp",
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": (
                    "\ud83d\udc84 Your Fashion Assistant\n"
                    "What would you like to do?\n"
                    "Choose an option \ud83d\udc47"
                )
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": "upload-wardrobe",
                            "title": "\ud83d\udce4 Upload Wardrobe",
                        },
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "ask-fashion-advice",
                            "title": "\ud83d\udcdc Ask Fashion Advice",
                        },
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "outfit-suggestion",
                            "title": "\ud83d\udc5a Outfit Suggestion",
                        },
                    },
                ]
            },
        },
    }


class WhatsAppTransport:
    """Minimal transport for sending WhatsApp Cloud API messages."""

    def __init__(
        self,
        *,
        phone_number_id: str,
        access_token: str,
        api_version: str = "v20.0",
        session: Optional[requests.Session] = None,
    ) -> None:
        self.phone_number_id = phone_number_id
        self.access_token = access_token
        self.api_version = api_version
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls) -> "WhatsAppTransport":
        """Create a transport using standard WhatsApp Cloud env vars."""

        phone_number_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
        access_token = os.environ.get("WHATSAPP_ACCESS_TOKEN")
        if not phone_number_id or not access_token:
            raise RuntimeError(
                "WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_ACCESS_TOKEN must be set."
            )
        return cls(phone_number_id=phone_number_id, access_token=access_token)

    def _post(self, payload: dict) -> dict:
        url = (
            f"https://graph.facebook.com/"
            f"{self.api_version}/{self.phone_number_id}/messages"
        )
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        response = self._session.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    def send_text(self, to: str, body: str) -> dict:
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"body": body, "preview_url": False},
        }
        return self._post(payload)

    def send_quick_reply_menu(self, to: str) -> dict:
        payload = build_quick_reply_menu()
        payload["to"] = to
        payload["recipient_type"] = "individual"
        return self._post(payload)


class WhatsAppStylistBot:
    """Routes WhatsApp webhook events to the AI stylist and sends replies."""

    def __init__(
        self,
        *,
        stylist: Optional[AIChatStylist] = None,
        transport: Optional[WhatsAppTransport] = None,
        menu_triggers: Optional[Iterable[str]] = None,
    ) -> None:
        self.stylist = stylist or AIChatStylist()
        self.transport = transport
        triggers = menu_triggers or ("menu", "start", "hi", "hello", "help")
        self.menu_triggers = {trigger.lower() for trigger in triggers}

    def handle_payload(self, payload: dict) -> list[dict]:
        """Process an incoming webhook payload and send responses."""

        if self.transport is None:
            raise RuntimeError("WhatsApp transport is not configured.")
        responses: list[dict] = []
        for message in extract_whatsapp_messages(payload):
            normalized = message.body.strip().lower()
            if normalized in self.menu_triggers:
                responses.append(self.transport.send_quick_reply_menu(message.wa_id))
                continue
            reply = self.stylist.chat(message.body)
            responses.append(self.transport.send_text(message.wa_id, reply))
        return responses


def build_whatsapp_webhook_response(payload: dict) -> List[dict]:
    """Convenience wrapper using env configured transport and stylist."""

    bot = WhatsAppStylistBot(transport=WhatsAppTransport.from_env())
    return bot.handle_payload(payload)
