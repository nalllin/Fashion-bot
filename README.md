# Fashion-bot

## Overview

This repository now contains an end-to-end, AI-driven wardrobe management platform. The
application exposes services for wardrobe creation, personalised outfit suggestions, a
text-and-image aware stylist chatbot, automated outfit visualisation, and WhatsApp menu
integration.

### Key Capabilities

1. **AI Fashion Recommendation Engine** – user garments are embedded with
   `text-embedding-3-small` (via OpenAI) or Sentence Transformers and indexed inside a
   FAISS vector store for similarity based pairing recommendations.
2. **AI Chat Stylist (Text + Image)** – a GPT-4o-mini powered conversational agent that
   pulls wardrobe context from the vector store to deliver personalised styling advice.
3. **AI Outfit Visualisation** – Stable Diffusion XL is used to generate outfit imagery
   using prompts engineered by GPT-4o-mini.
4. **WhatsApp Integration** – the WhatsApp client publishes menu options such as Create
   Wardrobe, Get Outfit Suggestions, and Chat with Stylist via the Cloud API.

## Project Structure

```
src/
  fashion_bot/
    api/                # FastAPI entry point exposing wardrobe, suggestions, chat and visualise endpoints
    embeddings/         # Text and image embedding adapters (OpenAI, Sentence Transformers, CLIP)
    integrations/       # WhatsApp Cloud API wrapper
    services/           # Wardrobe management, stylist chat, and outfit visualisation services
    vectorstores/       # FAISS vector store helper and metadata record definitions
```

## Running the API

1. Install dependencies (FastAPI, uvicorn, httpx, pillow, faiss-cpu, sentence-transformers,
   transformers, diffusers, torch, openai).
2. Export your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Start the FastAPI app:

   ```bash
   uvicorn fashion_bot.api.main:app --reload
   ```

## WhatsApp Integration

Instantiate the WhatsApp client with the Cloud API credentials and invoke `send_menu` to
display the primary actions in chat. Use incoming webhook payloads to route user
selections to the relevant FastAPI endpoints.

## Stable Diffusion Visualisation

The `OutfitVisualizer` wraps the Stable Diffusion XL pipeline. Provide a user-specific
style brief to the `PromptEngineer` to build descriptive prompts that emphasise garments,
colours, and ambience before generating the corresponding imagery.
