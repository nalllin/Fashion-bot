# Fashion-bot

Conversational AI stylist that blends text and image understanding.

## Features
- **GPT-4o-mini conversations**: natural back-and-forth styling guidance.
- **GPT-4o vision image intake**: summarize outfit photos into reusable context.
- **LangChain context management**: ChatPromptTemplate pipeline backed by FAISS retrieval memory.
- **Vector memory**: store prior wardrobe details with OpenAI embeddings.

## Getting Started
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Set the required OpenAI environment variables:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## Usage
Run the command line chat interface:
```bash
python -m ai_chat_stylist.app --image path/to/outfit.jpg
```
Type `exit` when you are done chatting.

### WhatsApp / Whati integration
1. Configure WhatsApp Cloud credentials (used by WhatI bots):
   ```bash
   export WHATSAPP_PHONE_NUMBER_ID="<phone-number-id>"
   export WHATSAPP_ACCESS_TOKEN="<permanent-token>"
   ```
2. In your WhatI bot webhook handler, import and forward payloads to the stylist:
   ```python
   from ai_chat_stylist.whatsapp_bot import build_whatsapp_webhook_response

   def handle_whati_event(payload: dict) -> None:
       # Sends quick reply menus for greetings and stylist responses for other input.
       build_whatsapp_webhook_response(payload)
   ```
3. To send the green quick-reply card shown in the mock, call
   `WhatsAppTransport.send_quick_reply_menu()` when the user sends "menu".

The helper automatically streams incoming WhatsApp/Whati text and button
messages into the `AIChatStylist` and posts the LLM responses back to WhatsApp so
the conversation continues inside the chat UI.

## Tests
```bash
pytest
```
