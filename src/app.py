from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from .ingest import DATA_DIR, ingest_pdfs
    from .model_config import get_chat_model_config, get_embedding_model_name, has_chat_api_key
    from .retriever import (
        format_sources,
        get_plain_llm,
        get_rag_chain,
        vectorstore_exists,
    )
    from .vectorstore_utils import VectorstoreAccessError, close_chroma_client
except ImportError:
    from ingest import DATA_DIR, ingest_pdfs
    from model_config import get_chat_model_config, get_embedding_model_name, has_chat_api_key
    from retriever import (
        format_sources,
        get_plain_llm,
        get_rag_chain,
        vectorstore_exists,
    )
    from vectorstore_utils import VectorstoreAccessError, close_chroma_client


st.set_page_config(page_title="RAG Document Intelligence", layout="wide")

OPENROUTER_FREE_MODELS_URL = "https://openrouter.ai/models?max_price=0"


@st.cache_resource(show_spinner=False)
def cached_rag_chain() -> Any:
    return get_rag_chain()


@st.cache_resource(show_spinner=False)
def cached_plain_llm() -> Any:
    return get_plain_llm()


def release_cached_chroma_resources() -> None:
    """Release cached Chroma clients before rebuilding the persisted store."""
    try:
        if has_chat_api_key() and vectorstore_exists():
            chain = cached_rag_chain()
            retriever = getattr(chain, "retriever", None)
            vector_store = getattr(retriever, "vectorstore", None)
            if vector_store is not None:
                close_chroma_client(vector_store)
    except Exception:
        # Reingest will surface the actionable error if the store is locked/corrupt.
        pass
    finally:
        st.cache_resource.clear()
        gc.collect()


def format_question_error(exc: Exception) -> str:
    message = str(exc)
    error_text = f"Could not answer the question: {message}"
    lower_message = message.lower()
    if "no endpoints found" in lower_message or (
        "error code: 404" in lower_message and "openrouter" in lower_message
    ):
        return (
            f"{error_text}\n\n"
            "Hint: update OPENROUTER_MODEL to a model currently listed in "
            f"OpenRouter Models ({OPENROUTER_FREE_MODELS_URL}), then restart Streamlit."
        )
    return error_text


def save_uploaded_pdfs(uploaded_files: list[Any]) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        if not filename.lower().endswith(".pdf"):
            continue
        target_path = DATA_DIR / filename
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)
    return saved_paths


def run_comparison(question: str) -> dict[str, Any]:
    rag_result = cached_rag_chain().invoke({"query": question})
    plain_response = cached_plain_llm().invoke(question)
    return {
        "question": question,
        "rag_answer": rag_result.get("result", ""),
        "plain_answer": str(getattr(plain_response, "content", plain_response)),
        "sources": format_sources(rag_result.get("source_documents", [])),
    }


def render_comparison(record: dict[str, Any]) -> None:
    rag_column, plain_column = st.columns(2)

    with rag_column:
        st.subheader("RAG answer")
        st.write(record["rag_answer"] or "No answer returned.")

        sources = record.get("sources", [])
        if sources:
            st.markdown("**Sources**")
            for source in sources:
                with st.expander(source["label"]):
                    st.write(source["excerpt"])
        else:
            st.caption("No source documents were returned.")

    with plain_column:
        st.subheader("Plain LLM answer")
        st.write(record["plain_answer"] or "No answer returned.")
        st.caption("This answer does not use your document vectorstore.")


def render_sidebar() -> tuple[bool, bool, str | None]:
    with st.sidebar:
        st.header("Documents")
        st.caption("Upload PDFs, build the Chroma vectorstore, then ask questions.")
        st.caption(f"Embeddings: local/free `{get_embedding_model_name()}`")

        chat_error: str | None = None
        try:
            chat_config = get_chat_model_config()
            chat_ready = has_chat_api_key(chat_config)
            st.info(f"LLM: {chat_config.display_name}")
            if chat_ready:
                st.success(f"{chat_config.api_key_env} detected.")
            else:
                chat_error = chat_config.missing_key_message
                st.error(chat_error)
        except RuntimeError as exc:
            chat_ready = False
            chat_error = str(exc)
            st.error(chat_error)


        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pdf_count = len(list(DATA_DIR.glob("*.pdf")))
        st.write(f"PDFs in data/: {pdf_count}")

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )

        ingest_disabled = not uploaded_files and pdf_count == 0
        if st.button("Save and ingest PDFs", disabled=ingest_disabled):
            try:
                saved_paths = save_uploaded_pdfs(uploaded_files or [])
                release_cached_chroma_resources()
                stats = ingest_pdfs()
                release_cached_chroma_resources()
                st.success(
                    "Ingested {pdf_files} PDF(s) into {chunks} chunks with local embeddings.".format(
                        **stats
                    )
                )
                if saved_paths:
                    st.caption("Saved: " + ", ".join(path.name for path in saved_paths))
            except VectorstoreAccessError as exc:
                st.error(f"Could not ingest PDFs: {exc}")
            except Exception as exc:
                st.error(f"Could not ingest PDFs: {exc}")

        vectorstore_error: VectorstoreAccessError | None = None
        try:
            vectorstore_ready = vectorstore_exists()
        except VectorstoreAccessError as exc:
            vectorstore_ready = False
            vectorstore_error = exc
            st.error(str(exc))
        if vectorstore_ready:
            st.success("Vectorstore ready.")
        elif vectorstore_error is not None:
            st.warning("Vectorstore unavailable until the access issue is resolved.")
        else:
            st.warning("Vectorstore missing. Ingest PDFs before asking questions.")

        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    return chat_ready, vectorstore_ready, chat_error


st.title("RAG Document Intelligence")
st.write(
    "Ask questions about your uploaded PDFs and compare a source-grounded RAG answer "
    "with a plain LLM answer."
)
try:
    current_chat_config = get_chat_model_config()
    st.caption(
        f"LLM: {current_chat_config.display_name} | "
        f"Embeddings: local/free `{get_embedding_model_name()}`"
    )
except RuntimeError:
    st.caption(f"Embeddings: local/free `{get_embedding_model_name()}`")

chat_ready, vectorstore_ready, chat_error = render_sidebar()

if not chat_ready:
    st.error(chat_error or "Configure the selected chat provider API key before using the chat.")
if not vectorstore_ready:
    st.info("Upload PDFs in the sidebar or place them in data/, then click Save and ingest PDFs.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(message["question"])
    with st.chat_message("assistant"):
        render_comparison(message)

chat_disabled = not chat_ready or not vectorstore_ready
question = st.chat_input("Ask a question about your PDFs", disabled=chat_disabled)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Comparing RAG and plain LLM answers..."):
            try:
                comparison = run_comparison(question)
                render_comparison(comparison)
                st.session_state.messages.append(comparison)
            except Exception as exc:
                st.error(format_question_error(exc))
