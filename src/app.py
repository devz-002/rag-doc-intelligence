from __future__ import annotations

import gc
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from .ingest import (
        DATA_DIR,
        PDF_TEXT_GUIDANCE,
        VECTORSTORE_DIR,
        ingest_pdfs,
        inspect_pdf_text,
        pdf_text_issue_message,
    )
    from .model_config import get_chat_model_config, get_embedding_model_name, has_chat_api_key
    from .retriever import (
        format_sources,
        get_plain_llm,
        get_rag_chain,
        vectorstore_exists,
    )
    from .vectorstore_utils import VectorstoreAccessError, close_chroma_client
except ImportError:
    from ingest import (
        DATA_DIR,
        PDF_TEXT_GUIDANCE,
        VECTORSTORE_DIR,
        ingest_pdfs,
        inspect_pdf_text,
        pdf_text_issue_message,
    )
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
DEFAULT_MAX_PDF_UPLOAD_MB = 100
SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


MAX_PDF_UPLOAD_MB = positive_int_env("MAX_PDF_UPLOAD_MB", DEFAULT_MAX_PDF_UPLOAD_MB)
MAX_PDF_UPLOAD_BYTES = MAX_PDF_UPLOAD_MB * 1024 * 1024
STREAMLIT_MAX_UPLOAD_MB = positive_int_env("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", 200)


@dataclass(frozen=True)
class UploadSaveResult:
    saved_paths: list[Path]
    errors: list[str]
    warnings: list[str]


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


def ensure_runtime_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


def safe_pdf_filename(filename: str) -> str:
    raw_name = Path(filename or "uploaded.pdf").name
    path_name = Path(raw_name)
    stem = SAFE_FILENAME_CHARS.sub("_", path_name.stem).strip("._") or "uploaded"
    return f"{stem}.pdf"


def next_available_path(directory: Path, filename: str) -> Path:
    target_path = directory / filename
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def validate_uploaded_pdf(uploaded_file: Any) -> str | None:
    filename = Path(getattr(uploaded_file, "name", "") or "").name or "uploaded file"
    if not filename.lower().endswith(".pdf"):
        return f"{filename}: only .pdf files can be ingested."

    size = getattr(uploaded_file, "size", None)
    if size == 0:
        return f"{filename}: the file is empty."
    if isinstance(size, int) and size > MAX_PDF_UPLOAD_BYTES:
        return (
            f"{filename}: file is {size / (1024 * 1024):.1f} MB; "
            f"the per-file limit is {MAX_PDF_UPLOAD_MB} MB."
        )

    try:
        header = bytes(uploaded_file.getbuffer()[:5])
    except Exception as exc:
        return f"{filename}: could not read the upload buffer ({exc})."
    if header != b"%PDF-":
        return f"{filename}: this does not look like a valid PDF file."

    inspection = inspect_pdf_text(BytesIO(uploaded_file.getbuffer()), source_name=filename)
    issue = pdf_text_issue_message(inspection)
    if issue is not None:
        return issue

    return None


def save_uploaded_pdfs(uploaded_files: list[Any]) -> UploadSaveResult:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    errors: list[str] = []
    warnings: list[str] = []
    for uploaded_file in uploaded_files:
        validation_error = validate_uploaded_pdf(uploaded_file)
        if validation_error:
            if "no selectable text found" in validation_error.lower():
                warnings.append(validation_error)
            else:
                errors.append(validation_error)
            continue
        filename = safe_pdf_filename(uploaded_file.name)
        target_path = next_available_path(DATA_DIR, filename)
        try:
            with target_path.open("wb") as output_file:
                output_file.write(uploaded_file.getbuffer())
        except OSError as exc:
            errors.append(f"{filename}: could not save to {DATA_DIR} ({exc}).")
            continue
        saved_paths.append(target_path)
    return UploadSaveResult(saved_paths=saved_paths, errors=errors, warnings=warnings)


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


        try:
            ensure_runtime_directories()
            pdf_count = len(list(DATA_DIR.glob("*.pdf")))
        except OSError as exc:
            pdf_count = 0
            st.error(f"Could not create runtime directories: {exc}")

        st.write(f"PDFs in `{DATA_DIR.name}/`: {pdf_count}")

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
        st.caption(
            f"Accepted PDFs up to {MAX_PDF_UPLOAD_MB} MB each. "
            f"If an upload card turns red, Streamlit rejected it before the app could save it; "
            f"try a valid PDF below {min(MAX_PDF_UPLOAD_MB, STREAMLIT_MAX_UPLOAD_MB)} MB."
        )
        st.caption(
            f"{PDF_TEXT_GUIDANCE} This app intentionally avoids heavy OCR dependencies for "
            "Hugging Face Docker Spaces."
        )

        if st.button("Save and ingest PDFs"):
            try:
                upload_result = save_uploaded_pdfs(uploaded_files or [])
                for error in upload_result.errors:
                    st.error(error)
                for warning in upload_result.warnings:
                    st.warning(warning)

                has_pdfs_to_ingest = pdf_count > 0 or bool(upload_result.saved_paths)
                if not has_pdfs_to_ingest:
                    st.error(
                        "No accepted PDFs are available to ingest. "
                        "Check the messages above, then upload a valid searchable PDF below "
                        f"{min(MAX_PDF_UPLOAD_MB, STREAMLIT_MAX_UPLOAD_MB)} MB."
                    )
                else:
                    release_cached_chroma_resources()
                    stats = ingest_pdfs()
                    release_cached_chroma_resources()
                    st.success(
                        "Ingested {pdf_files} PDF(s) into {chunks} chunks with local embeddings.".format(
                            **stats
                        )
                    )
                    for warning in stats.get("warnings", []):
                        st.warning(warning)
                    if upload_result.saved_paths:
                        st.caption(
                            "Saved: " + ", ".join(path.name for path in upload_result.saved_paths)
                        )
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
