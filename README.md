# Fashion-bot

Conversational AI stylist that blends text and image understanding.

## Features
- **GPT-4o-mini conversations**: natural back-and-forth styling guidance.
- **GPT-4o vision image intake**: summarize outfit photos into reusable context.
- **LangChain context management**: ChatPromptTemplate pipeline backed by FAISS retrieval memory.
- **Vector memory**: store prior wardrobe details with OpenAI embeddings.

## Getting Started

### Local virtual environment (recommended)
1. Create the project virtual environment in the repository root and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Set the required OpenAI environment variables in the same shell session:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

### Running from Google Colab
If you prefer to use your Colab GPU credits, you can clone the repository inside a Colab notebook and reuse the same setup commands:
1. Start a new notebook and run:
   ```python
   !git clone https://github.com/<your-account>/Fashion-bot.git
   %cd Fashion-bot
   !python -m venv .venv
   !source .venv/bin/activate
   !pip install -r requirements.txt
   ```
   > **Tip:** Colab cells run in separate shells. To avoid re-activating the environment in every cell, you can install dependencies directly into the notebook kernel by prefixing commands with `pip` instead of creating a venv.
2. Add your OpenAI API key to Colab:
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "sk-..."
   ```
3. Launch the CLI module or call the chatbot classes directly from new cells.

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

## Implementation & Testing Notebook

A step-by-step walkthrough lives at [`notebooks/AIChatStylist_demo.ipynb`](notebooks/AIChatStylist_demo.ipynb). It shows how to:

- Configure local or Colab environments and swap between live OpenAI usage and the built-in offline demo mode.
- Instantiate the `AIChatStylist`, optionally ingest outfit imagery, and inspect the stored FAISS context.
- Trigger the full `pytest` suite directly from the notebook to validate the pipeline end-to-end.

Open the notebook in Jupyter, VS Code, or Colab to explore the implementation interactively.
