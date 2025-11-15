from __future__ import annotations

from ai_chat_stylist.whatsapp_bot import (
    WatiStylistBot,
    build_quick_reply_menu_payload,
    extract_wati_messages,
)


def sample_payload(message_type: str, body: str) -> dict:
    message: dict = {
        "from": "15551234567",
        "id": "wamid.ABCD",
        "type": message_type,
    }
    if message_type == "text":
        message["text"] = {"body": body}
    else:
        message["interactive"] = {
            "type": "button_reply",
            "button_reply": {"title": body},
        }
    return {"messages": [message]}


def test_extract_wati_messages_supports_text_and_buttons():
    text_payload = sample_payload("text", "Ask Fashion Advice")
    interactive_payload = sample_payload("interactive", "Outfit Suggestion")

    text_messages = extract_wati_messages(text_payload)
    button_messages = extract_wati_messages(interactive_payload)

    assert text_messages[0].body == "Ask Fashion Advice"
    assert button_messages[0].body == "Outfit Suggestion"


def test_quick_reply_menu_matches_expected_structure():
    menu = build_quick_reply_menu_payload()
    assert menu["header"]["text"].startswith("\ud83d\udc84")
    assert len(menu["buttons"]) == 3


class DummyStylist:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def chat(self, message: str) -> str:
        self.seen.append(message)
        return f"LLM: {message}"


class DummyTransport:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send_text(self, to: str, body: str) -> dict:
        self.sent.append((to, body))
        return {"id": len(self.sent)}

    def send_quick_reply_menu(self, to: str) -> dict:
        self.sent.append((to, "menu"))
        return {"id": len(self.sent)}


def test_bot_routes_messages_and_handles_menu_trigger():
    stylist = DummyStylist()
    transport = DummyTransport()
    bot = WatiStylistBot(stylist=stylist, transport=transport)

    hi_payload = sample_payload("text", "Hi")
    question_payload = sample_payload("text", "What goes with chinos?")

    bot.handle_payload(hi_payload)
    bot.handle_payload(question_payload)

    assert transport.sent[0][1] == "menu"
    assert transport.sent[1][1].startswith("LLM: What goes with chinos?")
    assert stylist.seen == ["What goes with chinos?"]
