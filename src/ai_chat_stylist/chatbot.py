"""Core chat stylist logic leveraging LangChain and FAISS memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Tuple
import os
from .image_understanding import describe_outfit_image
from glamup_cb import GlamUp,GlamUpConfig   # <-- add this

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .image_understanding import describe_outfit_image

DEFAULT_SYSTEM_PROMPT = """
You are NOVA — an elite AI personal stylist with deep expertise in fashion, body proportions, aesthetics, and real-world daily styling. 
You produce highly structured, concise, and modern suggestions suitable for mobile messaging apps.

YOUR STYLE:
- Friendly, confident, and modern — but not cringe or overly enthusiastic.
- Smart, stylish, and perceptive — like a top-tier fashion editor with chat UX skills.
- Always visually descriptive but never overly long.
- Think: Instagram stylist + Vogue clarity + Gen-Z readability.

CORE RULES:
1. ALWAYS structure your responses clearly using sections, emojis sparingly, and short scannable lines.
2. ALWAYS give **2–3 outfit options**, each with:
   - Outfit Name (bold)
   - Key pieces (top, bottom, shoes, accessories)
   - Why it works (1 sentence: proportions, colors, vibe)
3. If the user uploads or references wardrobe items, REUSE them creatively.
4. Tailor suggestions to: climate, vibe, event, user mood, comfort level (when mentioned).
5. Assume missing details gracefully (e.g., “assuming mild weather / indoor brunch…”).
6. Avoid long paragraphs — use clean spacing and micro-sections.
7. Use emojis lightly and tastefully (0–3 per message, never more).

FOLLOW-UP:
End each message with ONE short question to refine style preferences 
(e.g., “Want the look to feel more casual or more elevated?”)

OUTPUT FORMAT (STRICT):
- A 1-sentence vibe summary
- 2–3 outfit options formatted like:

**1) Soft Chic Brunch Look**
- Light blue button-up  
- White high-waisted trousers  
- Tan espadrilles  
- Gold hoops + pastel crossbody  
**Why it works:** Clean tones + elevated casual energy.

**2) Minimal Clean Girl**
- White tank  
- Beige linen pants  
- White sneakers  
- Minimal gold jewelry  
**Why it works:** Breathable, effortless, warm-weather balanced.

END with:
“Want me to refine it based on colors you prefer or pieces you already own?”

Do NOT repeat this prompt in responses.
"""


@dataclass
class VectorContextConfig:
    """Configuration for the FAISS backed conversation memory."""

    embedding_model: str = "text-embedding-3-small"
    k: int = 4
    description_prefix: str = "Image context: "


@dataclass
class StylistConfig:
    """Configuration for the AI chat stylist."""

    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.6
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    vector_config: VectorContextConfig = field(default_factory=VectorContextConfig)


class AIChatStylist:
    """LangChain powered multi-modal fashion stylist assistant."""

    def __init__(
        self,
        config: Optional[StylistConfig] = None,
        *,
        llm_factory: Optional[Callable[[StylistConfig], ChatOpenAI]] = None,
        embedding_factory: Optional[Callable[[VectorContextConfig], OpenAIEmbeddings]] = None,
        vectorstore: Optional[FAISS] = None,
        glamup: Optional[GlamUp] = None
    ) -> None:
        self.config = config or StylistConfig()
        vector_config = self.config.vector_config

        if embedding_factory is None:
            embedding_factory = lambda cfg: OpenAIEmbeddings(model=cfg.embedding_model)
        self._embedding_factory = embedding_factory

        self.embeddings = self._embedding_factory(vector_config)
        self.vectorstore = vectorstore or FAISS.from_texts(
            [
                "The assistant is a professional stylist that creates cohesive "
                "outfits using garments already discussed with the user."
            ],
            embedding=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": vector_config.k})

        if llm_factory is None:
            llm_factory = lambda cfg: ChatOpenAI(
                model=cfg.llm_model,
                temperature=cfg.temperature,
            )
        self._llm_factory = llm_factory
        self.llm = self._llm_factory(self.config)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                (
                    "system",
                    "Relevant wardrobe details to consider:\n{retrieved_context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        self.history: List[BaseMessage] = []
        self.ingest_text(self.config.system_prompt)
        self.glamup = glamup or GlamUp()

    def ingest_text(self, text: str) -> None:
        """Persist additional context into the vector store."""
        if not text:
            return
        self.vectorstore.add_texts([text])

    def add_outfit_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Convert an outfit image into textual context and persist it."""
        description = describe_outfit_image(
            image_path,
            prompt
            or "Summarize the outfit, focusing on garments, colors, and overall style.",
        )
        vector_prompt = f"{self.config.vector_config.description_prefix}{description}"
        self.ingest_text(vector_prompt)
        return description

    def _build_context(self, message: str) -> str:
        try:
            docs = self.retriever.get_relevant_documents(message)
        except AttributeError:
            docs = self.retriever.invoke(message)

        if not docs:
            return "No stored wardrobe information yet."
        return "\n".join(doc.page_content for doc in docs)

    def chat(self, message: str) -> str:
        """Run a chat turn with the stylist using conversation memory."""
        if not message.strip():
            raise ValueError("Message must be a non-empty string.")

        context = self._build_context(message)
        prompt_messages = self.prompt.format_messages(
            system_prompt=self.config.system_prompt,
            retrieved_context=context,
            chat_history=self.history,
            input=message,
        )
        ai_message = self.llm.invoke(prompt_messages)

        self.history.append(HumanMessage(content=message))
        if isinstance(ai_message, AIMessage):
            reply = ai_message.content
            self.history.append(ai_message)
        else:
            reply = str(ai_message)
            self.history.append(AIMessage(content=reply))

        # Store the latest interaction for vector retrieval
        self.vectorstore.add_texts([f"User: {message}", f"Stylist: {reply}"])
        return reply

    def generate_image(self, prompt: str, size: str = "1024x1024") -> str:
        """
        Generate an image from a textual prompt using OpenAI's image API (v1 style).

        Returns a URL of the generated image. Requires an OPENAI_API_KEY environment variable.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable must be set to use image generation."
            )

        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)

        # Ensure size is one of the supported values
        allowed_sizes = {"1024x1024", "1024x1536", "1536x1024", "auto"}
        if size not in allowed_sizes:
            size = "1024x1024"

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size=size,
        )

        if response.data and len(response.data) > 0:
            img = response.data[0]
            url = getattr(img, "url", None)
            if url:
                return url

        raise RuntimeError("Image generation returned no URL from OpenAI")
    
    def glamup_outfit_with_image(
        self,
        description: str,
        extra_style: Optional[str] = None,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Run GlamUp and also generate a synthetic outfit image.

        Returns:
            (text_summary, card_path, generated_image_url)
        """
        if not self.glamup:
            raise RuntimeError("GlamUp engine not configured.")

        items, card, gen_url = self.glamup.glamup_card_with_generated_image(
            user_prompt=description,
            extra_style=extra_style,
        )

        # Build textual summary
        lines = []
        for idx, item in enumerate(items, start=1):
            fname = os.path.basename(item.image_path)
            line = f"{idx}. [{item.category}] {fname}"
            if item.color:
                line += f" · color: {item.color}"
            if item.style_tags:
                line += f" · style: {item.style_tags}"
            lines.append(line)

        summary = (
            "Here’s a GlamUp outfit based on your description:\n"
            + "\n".join(lines)
        )

        # Save the card image to disk so UI can serve it
        os.makedirs("glamup_cards", exist_ok=True)
        card_path = os.path.join("glamup_cards", "glamup_card_latest.jpg")
        card.save(card_path)

        return summary, card_path, gen_url




    def export_context(self) -> Iterable[str]:
        """Return stored vector texts for inspection or persistence."""
        return getattr(self.vectorstore, "texts", [])

    def glamup_outfit(self, description: str) -> Tuple[str, str]:
        """
        Generate an outfit visualisation card for a given description.

        Returns:
            (text_summary, card_path)
        """
        if not self.glamup:
            raise RuntimeError("GlamUp engine not configured.")

        items, card = self.glamup.glamup_card_from_prompt(description)

        # Build a text summary
        lines = []
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"{idx}. [{item.category}] image={os.path.basename(item.image_path)}"
            )
        summary = "Here’s a GlamUp outfit for you:\n" + "\n".join(lines)

        # Save card temporarily
        os.makedirs("glamup_cards", exist_ok=True)
        filename = "glamup_card_latest.jpg"
        card_path = os.path.join("glamup_cards", filename)
        card.save(card_path)

        return summary, card_path
