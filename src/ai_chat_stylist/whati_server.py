"""Tiny HTTP server that connects the WATI webhook to the stylist."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar, Tuple

from .whatsapp_bot import WatiStylistBot, WatiTransport


class WatiWebhookHandler(BaseHTTPRequestHandler):
    """Basic handler that accepts POST webhooks from WATI."""

    bot: ClassVar[WatiStylistBot | None] = None

    @classmethod
    def configure(cls, bot: WatiStylistBot) -> None:
        cls.bot = bot

    def _json_response(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._json_response(200, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(data or b"{}")
        except json.JSONDecodeError:
            self._json_response(400, {"error": "Invalid JSON"})
            return
        bot = self.bot
        if bot is None:
            bot = WatiStylistBot(transport=WatiTransport.from_env())
            self.configure(bot)
        try:
            responses = bot.handle_payload(payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._json_response(500, {"error": str(exc)})
            return
        self._json_response(200, {"status": "ok", "responses": responses})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        if os.environ.get("WATI_SERVER_LOG_REQUESTS", "1") == "1":
            super().log_message(format, *args)


def run(host: str | None = None, port: int | None = None) -> Tuple[str, int]:
    """Start a blocking HTTP server that processes WATI events."""
    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", "8080"))
    server = HTTPServer((host, port), WatiWebhookHandler)
    bot = WatiStylistBot(transport=WatiTransport.from_env())
    WatiWebhookHandler.configure(bot)
    print(f"WATI webhook server listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        print("Stopping WATI webhook server...")
    finally:
        server.server_close()
    return host, port


if __name__ == "__main__":
    run()
