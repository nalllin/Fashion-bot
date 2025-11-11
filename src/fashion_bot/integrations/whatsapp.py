"""WhatsApp integration layer using a generic REST API."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import httpx


@dataclass
class WhatsAppConfig:
    """Configuration required to call the WhatsApp Cloud API or similar gateways."""

    base_url: str
    access_token: str
    phone_number_id: str


@dataclass
class WhatsAppClient:
    """Thin wrapper that sends template messages and interactive menus."""

    config: WhatsAppConfig

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json",
        }

    def send_message(self, to: str, message: str) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": message},
        }
        response = httpx.post(
            f"{self.config.base_url}/{self.config.phone_number_id}/messages",
            headers=self._headers(),
            content=json.dumps(payload),
            timeout=10,
        )
        response.raise_for_status()

    def send_menu(self, to: str) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": "How can I help you today?"},
                "action": {
                    "buttons": [
                        {"type": "reply", "reply": {"id": "create_wardrobe", "title": "Create Wardrobe"}},
                        {"type": "reply", "reply": {"id": "get_suggestions", "title": "Get Outfit Suggestions"}},
                        {"type": "reply", "reply": {"id": "chat_stylist", "title": "Chat with Stylist"}},
                    ]
                },
            },
        }
        response = httpx.post(
            f"{self.config.base_url}/{self.config.phone_number_id}/messages",
            headers=self._headers(),
            content=json.dumps(payload),
            timeout=10,
        )
        response.raise_for_status()

    def mark_message_read(self, message_id: str) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        response = httpx.post(
            f"{self.config.base_url}/{self.config.phone_number_id}/messages",
            headers=self._headers(),
            content=json.dumps(payload),
            timeout=10,
        )
        response.raise_for_status()
