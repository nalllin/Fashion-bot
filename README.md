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
   These **are the only secrets the bridge code needs**. Add them to the shell
   that runs your webhook server (for local testing) or to your production
   secrets manager/environment variables.
2. Expose a webhook endpoint (Flask/FastAPI/Cloud Function) that WhatI can call
   and forward the payload to the helper:
   ```python
   from ai_chat_stylist.whatsapp_bot import build_whatsapp_webhook_response

   def handle_whati_event(payload: dict) -> None:
       # Sends quick reply menus for greetings and stylist responses for other input.
       build_whatsapp_webhook_response(payload)
   ```
   *Your WhatsApp Cloud App verification challenge handler lives alongside this
   endpoint; once Meta verifies the URL, configure the same https URL inside the
   WhatI flow as the webhook.*
3. Configure WhatI:
   - **Webhook URL** – point it at the endpoint you created in step 2.
   - **Credentials** – supply the same `WHATSAPP_PHONE_NUMBER_ID` and
     `WHATSAPP_ACCESS_TOKEN` inside WhatI if it needs to call the Cloud API on
     your behalf (otherwise the environment variables above cover it).
   - **Triggers** – route text/button replies from WhatsApp to the webhook so
     they arrive in the payload consumed by `build_whatsapp_webhook_response`.
4. To send the green quick-reply card shown in the mock, call
   `WhatsAppTransport.send_quick_reply_menu()` when the user sends "menu". The
   helper automatically streams incoming WhatsApp/Whati text and button messages
   into the `AIChatStylist` and posts the LLM responses back to WhatsApp so the
   conversation continues inside the chat UI.

## Tests
```bash
pytest
```
