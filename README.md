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

## Tests
```bash
pytest
```
