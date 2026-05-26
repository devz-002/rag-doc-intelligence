from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EMBEDDING_MODEL = "chroma-default-onnx-minilm-l6-v2"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
SUPPORTED_LLM_PROVIDERS = {"openai", "openrouter"}
SUPPORTED_EMBEDDING_MODEL_ALIASES = {
    DEFAULT_EMBEDDING_MODEL,
    "chroma/onnx-minilm-l6-v2",
    "onnx-mini-lm-l6-v2",
    "onnx_mini_lm_l6_v2",
    "sentence-transformers/all-minilm-l6-v2",
}
PLACEHOLDER_API_KEYS = {
    "",
    "sk-your-key-here",
    "sk-your-real-key",
    "sk-or-your-key-here",
    "sk-or-v1-your-key-here",
    "your-openrouter-api-key",
    "your-openrouter-api-key-here",
}


class ChromaONNXMiniLMEmbeddings(Embeddings):
    """LangChain adapter for Chroma's local default ONNX MiniLM embeddings."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
            ONNXMiniLM_L6_V2,
        )

        self._embedding_function = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [
            self._to_float_list(embedding)
            for embedding in self._embedding_function(list(texts))
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @staticmethod
    def _to_float_list(embedding: Any) -> list[float]:
        values = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        return [float(value) for value in values]


@dataclass(frozen=True)
class ChatModelConfig:
    provider: str
    model: str
    api_key_env: str
    api_key: str
    base_url: str | None = None

    @property
    def provider_label(self) -> str:
        return "OpenRouter" if self.provider == "openrouter" else "OpenAI"

    @property
    def display_name(self) -> str:
        return f"{self.provider_label} / {self.model}"

    @property
    def has_api_key(self) -> bool:
        return self.api_key.strip() not in PLACEHOLDER_API_KEYS

    @property
    def missing_key_message(self) -> str:
        return (
            f"{self.provider_label} is selected but {self.api_key_env} is not configured. "
            "Copy .env.example to .env, set LLM_PROVIDER, and add the matching API key."
        )


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def get_embedding_model_name() -> str:
    load_project_env()
    requested_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
    if not requested_model:
        return DEFAULT_EMBEDDING_MODEL
    if requested_model.lower() in SUPPORTED_EMBEDDING_MODEL_ALIASES:
        return DEFAULT_EMBEDDING_MODEL
    return requested_model


def get_embeddings() -> Embeddings:
    """Return the local embedding model shared by ingestion and retrieval."""
    embedding_model = get_embedding_model_name()
    if embedding_model != DEFAULT_EMBEDDING_MODEL:
        raise RuntimeError(
            f"Unsupported EMBEDDING_MODEL={embedding_model!r}. "
            f"Use {DEFAULT_EMBEDDING_MODEL!r} for local Chroma ONNX MiniLM embeddings."
        )
    return ChromaONNXMiniLMEmbeddings()


def get_llm_provider() -> str:
    load_project_env()
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower() or "openai"
    if provider not in SUPPORTED_LLM_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
        raise RuntimeError(f"Unsupported LLM_PROVIDER={provider!r}. Use one of: {supported}.")
    return provider


def get_chat_model_config() -> ChatModelConfig:
    provider = get_llm_provider()
    if provider == "openrouter":
        return ChatModelConfig(
            provider=provider,
            model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
            or DEFAULT_OPENROUTER_MODEL,
            api_key_env="OPENROUTER_API_KEY",
            api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip()
            or DEFAULT_OPENROUTER_BASE_URL,
        )

    return ChatModelConfig(
        provider=provider,
        model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL,
        api_key_env="OPENAI_API_KEY",
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
    )


def has_chat_api_key(config: ChatModelConfig | None = None) -> bool:
    config = config or get_chat_model_config()
    return config.has_api_key


def require_chat_api_key() -> ChatModelConfig:
    config = get_chat_model_config()
    if not config.has_api_key:
        raise RuntimeError(config.missing_key_message)
    return config


def get_chat_model(*, temperature: float = 0) -> ChatOpenAI:
    config = require_chat_api_key()
    kwargs: dict[str, object] = {
        "model": config.model,
        "temperature": temperature,
        "api_key": config.api_key,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenAI(**kwargs)
