from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    from .model_config import get_embedding_model_name, get_embeddings
    from .vectorstore_utils import (
        VectorstoreAccessError,
        close_chroma_client,
        ensure_vectorstore_directory,
        is_vectorstore_access_error,
        vectorstore_access_message,
    )
except ImportError:
    from model_config import get_embedding_model_name, get_embeddings
    from vectorstore_utils import (
        VectorstoreAccessError,
        close_chroma_client,
        ensure_vectorstore_directory,
        is_vectorstore_access_error,
        vectorstore_access_message,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
COLLECTION_NAME = "rag_doc_intelligence"
PDF_TEXT_GUIDANCE = (
    "Scanned/image-only PDFs need OCR preprocessing first. "
    "Create a searchable PDF with OCR, then upload that OCR-processed file."
)


@dataclass(frozen=True)
class PdfTextInspection:
    """Lightweight pypdf diagnostics for deciding whether a PDF can be ingested."""

    source_name: str
    page_count: int | None = None
    is_encrypted: bool = False
    has_extractable_text: bool = False
    checked_pages: int = 0
    parse_error: str | None = None
    text_error: str | None = None


def inspect_pdf_text(
    source: Any,
    *,
    source_name: str | None = None,
    max_pages: int | None = None,
) -> PdfTextInspection:
    """Check page count, encryption, and whether pypdf can extract selectable text."""
    if source_name is None:
        source_name = Path(source).name if isinstance(source, (str, Path)) else "uploaded PDF"

    try:
        reader = PdfReader(source, strict=False)
    except Exception as exc:
        return PdfTextInspection(source_name=source_name, parse_error=str(exc))

    if reader.is_encrypted:
        return PdfTextInspection(source_name=source_name, is_encrypted=True)

    try:
        page_count = len(reader.pages)
    except Exception as exc:
        return PdfTextInspection(source_name=source_name, parse_error=str(exc))

    if page_count == 0:
        return PdfTextInspection(source_name=source_name, page_count=0)

    pages_to_check = page_count if max_pages is None else min(page_count, max_pages)
    first_text_error: str | None = None
    for page_index in range(pages_to_check):
        try:
            page_text = reader.pages[page_index].extract_text() or ""
        except Exception as exc:
            if first_text_error is None:
                first_text_error = str(exc)
            continue
        if page_text.strip():
            return PdfTextInspection(
                source_name=source_name,
                page_count=page_count,
                has_extractable_text=True,
                checked_pages=page_index + 1,
                text_error=first_text_error,
            )

    return PdfTextInspection(
        source_name=source_name,
        page_count=page_count,
        checked_pages=pages_to_check,
        text_error=first_text_error,
    )


def inspect_pdf_files(pdf_files: list[Path]) -> list[PdfTextInspection]:
    """Inspect saved PDFs before running the LangChain loader."""
    return [inspect_pdf_text(pdf_file, source_name=pdf_file.name) for pdf_file in pdf_files]


def pdf_text_issue_message(inspection: PdfTextInspection) -> str | None:
    if inspection.parse_error:
        return f"{inspection.source_name}: could not be parsed as a PDF ({inspection.parse_error})."
    if inspection.is_encrypted:
        return f"{inspection.source_name}: password-protected or encrypted PDFs are not supported."
    if inspection.page_count == 0:
        return f"{inspection.source_name}: no pages were found in the PDF."
    if not inspection.has_extractable_text:
        if inspection.page_count is None:
            page_scope = "the PDF"
        elif inspection.checked_pages < inspection.page_count:
            page_scope = f"the first {inspection.checked_pages} of {inspection.page_count} page(s)"
        else:
            page_scope = f"{inspection.checked_pages} page(s)"
        detail = f"{inspection.source_name}: no selectable text found in {page_scope}."
        if inspection.text_error:
            detail += f" Text extraction error: {inspection.text_error}."
        return f"{detail} {PDF_TEXT_GUIDANCE}"
    return None


def pdf_text_warnings(inspections: list[PdfTextInspection]) -> list[str]:
    return [
        message
        for message in (pdf_text_issue_message(inspection) for inspection in inspections)
        if message is not None
    ]


def no_text_ingest_error(
    data_path: Path,
    inspections: list[PdfTextInspection],
    reason: str,
) -> str:
    issues = pdf_text_warnings(inspections)
    details = "\n".join(f"- {issue}" for issue in issues)
    if details:
        return (
            f"{reason} Problem PDF diagnostics:\n{details}\n"
            "Try a PDF where text can be selected/copied, or OCR the file first and upload the "
            "searchable PDF."
        )
    return (
        f"{reason} PDF files were found in {data_path}, but no selectable text could be extracted. "
        f"{PDF_TEXT_GUIDANCE}"
    )


def ingest_pdfs(
    data_dir: Path | str = DATA_DIR,
    persist_directory: Path | str = VECTORSTORE_DIR,
    *,
    reset: bool = True,
) -> dict[str, Any]:
    """Load PDFs from data_dir, split them, embed them, and persist a Chroma index."""
    data_path = Path(data_dir)
    persist_path = Path(persist_directory)
    data_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_path}.")

    pdf_inspections = inspect_pdf_files(pdf_files)
    inspection_warnings = pdf_text_warnings(pdf_inspections)
    blockers: list[str] = []
    for inspection in pdf_inspections:
        if inspection.parse_error or inspection.is_encrypted or inspection.page_count == 0:
            issue = pdf_text_issue_message(inspection)
            if issue is not None:
                blockers.append(issue)
    if blockers:
        details = "\n".join(f"- {blocker}" for blocker in blockers)
        raise ValueError(f"Some PDFs cannot be read:\n{details}\nRemove or replace them, then ingest again.")

    if not any(inspection.has_extractable_text for inspection in pdf_inspections):
        raise ValueError(
            no_text_ingest_error(
                data_path,
                pdf_inspections,
                "No extractable text was found in the PDFs, so the vectorstore was not rebuilt.",
            )
        )

    loader = PyPDFDirectoryLoader(str(data_path))
    documents = loader.load()
    if not documents:
        raise ValueError(
            no_text_ingest_error(
                data_path,
                pdf_inspections,
                f"PDF files were found in {data_path}, but no page documents could be loaded.",
            )
        )

    text_documents = [
        document for document in documents if getattr(document, "page_content", "").strip()
    ]
    if not text_documents:
        raise ValueError(
            no_text_ingest_error(
                data_path,
                pdf_inspections,
                f"Loaded {len(documents)} page document(s), but none contained extractable text.",
            )
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(text_documents)
    if not chunks:
        raise ValueError(
            no_text_ingest_error(
                data_path,
                pdf_inspections,
                "No text chunks were created from the loaded PDFs.",
            )
        )

    embeddings = get_embeddings()
    vector_store = None
    try:
        ensure_vectorstore_directory(persist_path)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_path),
        )
        if reset:
            vector_store.reset_collection()
        vector_store.add_documents(chunks)

        persist = getattr(vector_store, "persist", None)
        if callable(persist):
            persist()
    except VectorstoreAccessError:
        raise
    except Exception as exc:
        if is_vectorstore_access_error(exc):
            raise VectorstoreAccessError(vectorstore_access_message(persist_path, exc)) from exc
        raise
    finally:
        if vector_store is not None:
            close_chroma_client(vector_store)

    return {
        "pdf_files": len(pdf_files),
        "documents": len(documents),
        "text_documents": len(text_documents),
        "chunks": len(chunks),
        "warnings": inspection_warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chroma vectorstore from PDFs.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing PDFs.")
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=VECTORSTORE_DIR,
        help="Directory where Chroma should persist the index.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Append to the existing Chroma collection instead of rebuilding it.",
    )
    args = parser.parse_args()

    stats = ingest_pdfs(
        data_dir=args.data_dir,
        persist_directory=args.persist_directory,
        reset=not args.no_reset,
    )
    print(
        "Ingested {pdf_files} PDF(s), loaded {documents} page document(s) "
        "({text_documents} with extractable text), and stored {chunks} chunk(s).".format(**stats)
    )
    for warning in stats.get("warnings", []):
        print(f"Warning: {warning}")
    print(f"Embedding model: {get_embedding_model_name()} (local Chroma ONNX MiniLM)")


if __name__ == "__main__":
    main()
