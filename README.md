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

### WhatsApp / WATI (Whati) integration
1. Configure the WATI credentials (replace the examples with your real tenant
   values – the user-provided instance looks like `https://live-mt-server.wati.io/1048684`):
   ```bash
   export WATI_SERVER_URL="https://live-mt-server.wati.io/1048684"
   export WATI_API_KEY="<paste the Bearer token you received>"
   ```
   The helper automatically signs every request to the WATI REST API using these
   values. Never commit the token to source control – inject it through env vars
   wherever the webhook server runs.
2. Start the bundled webhook server (it uses Python's `http.server` so there are
   zero extra dependencies):
   ```bash
   python -m ai_chat_stylist.whati_server
   ```
   The server binds to `0.0.0.0:8080` by default. Override with `HOST`/`PORT`
   environment variables if you need a different interface.
3. Point the WATI/Whati webhook URL at the server you just launched. Every
   incoming payload gets routed through `WatiStylistBot`, which handles menu
   triggers and forwards other messages to `AIChatStylist`.
4. The quick-reply UI from the mock is implemented with WATI interactive
   buttons. Anytime the user types `hi`, `hello`, `menu`, `start`, or `help`,
   the bot calls the WATI "send interactive buttons" endpoint so that the three
   shortcut buttons show up inside WhatsApp. All other inputs produce LLM
   replies that the transport sends back via `sendSessionMessage`, keeping the
   conversation inside WhatsApp.

## Tests
```bash
pytest
```
