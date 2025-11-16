"""Core chat stylist logic leveraging LangChain and FAISS memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional
import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .image_understanding import describe_outfit_image

DEFAULT_SYSTEM_PROMPT = (
    "You are an upbeat and thoughtful personal stylist. You remember prior "
    "details from the conversation and refer back to the user's wardrobe when "
    "suggesting outfits. Always include a short explanation for each styling "
    "suggestion."
)


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



    def export_context(self) -> Iterable[str]:
        """Return stored vector texts for inspection or persistence."""
        return getattr(self.vectorstore, "texts", [])
