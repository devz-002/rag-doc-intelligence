from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_classic.chains import RetrievalQA
from langchain_chroma import Chroma

try:
    from .model_config import get_chat_model, get_embedding_model_name, get_embeddings
    from .vectorstore_utils import (
        VectorstoreAccessError,
        is_vectorstore_access_error,
        vectorstore_access_message,
    )
except ImportError:
    from model_config import get_chat_model, get_embedding_model_name, get_embeddings
    from vectorstore_utils import (
        VectorstoreAccessError,
        is_vectorstore_access_error,
        vectorstore_access_message,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
COLLECTION_NAME = "rag_doc_intelligence"


def vectorstore_exists(persist_directory: Path | str = VECTORSTORE_DIR) -> bool:
    persist_path = Path(persist_directory)
    try:
        return persist_path.is_dir() and any(persist_path.iterdir())
    except OSError as exc:
        if is_vectorstore_access_error(exc):
            raise VectorstoreAccessError(vectorstore_access_message(persist_path, exc)) from exc
        raise


def get_vectorstore(persist_directory: Path | str = VECTORSTORE_DIR) -> Chroma:
    persist_path = Path(persist_directory)
    if not vectorstore_exists(persist_path):
        raise FileNotFoundError(
            f"No Chroma vectorstore found at {persist_path}. "
            f"Run `python src/ingest.py` first with embeddings {get_embedding_model_name()!r}."
        )

    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_path),
        embedding_function=embeddings,
    )


def get_retriever(persist_directory: Path | str = VECTORSTORE_DIR) -> Any:
    vector_store = get_vectorstore(persist_directory)
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})


def get_rag_chain(persist_directory: Path | str = VECTORSTORE_DIR) -> RetrievalQA:
    retriever = get_retriever(persist_directory)
    llm = get_chat_model(temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


def get_plain_llm() -> Any:
    return get_chat_model(temperature=0)


def answer_with_rag(question: str, persist_directory: Path | str = VECTORSTORE_DIR) -> dict[str, Any]:
    chain = get_rag_chain(persist_directory)
    result = chain.invoke({"query": question})
    return {
        "answer": result.get("result", ""),
        "source_documents": result.get("source_documents", []),
    }


def answer_with_plain_gpt(question: str) -> str:
    response = get_plain_llm().invoke(question)
    return str(getattr(response, "content", response))


def format_sources(source_documents: list[Any]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for index, document in enumerate(source_documents, start=1):
        metadata = getattr(document, "metadata", {}) or {}
        source = Path(str(metadata.get("source", "Unknown source"))).name
        page = metadata.get("page")
        if isinstance(page, int):
            location = f"{source}, page {page + 1}"
        elif page is not None:
            location = f"{source}, page {page}"
        else:
            location = source

        text = " ".join(getattr(document, "page_content", "").split())
        formatted.append(
            {
                "label": f"[{index}] {location}",
                "excerpt": text[:500] + ("..." if len(text) > 500 else ""),
            }
        )
    return formatted
