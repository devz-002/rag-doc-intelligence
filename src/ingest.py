from __future__ import annotations

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def ingest_pdfs(
    data_dir: Path | str = DATA_DIR,
    persist_directory: Path | str = VECTORSTORE_DIR,
    *,
    reset: bool = True,
) -> dict[str, int]:
    """Load PDFs from data_dir, split them, embed them, and persist a Chroma index."""
    data_path = Path(data_dir)
    persist_path = Path(persist_directory)
    data_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_path}.")

    loader = PyPDFDirectoryLoader(str(data_path))
    documents = loader.load()
    if not documents:
        raise ValueError(f"PDF files were found in {data_path}, but no text could be loaded.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were created from the loaded PDFs.")

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
        "chunks": len(chunks),
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
        "Ingested {pdf_files} PDF(s), loaded {documents} page document(s), "
        "and stored {chunks} chunk(s).".format(**stats)
    )
    print(f"Embedding model: {get_embedding_model_name()} (local Chroma ONNX MiniLM)")


if __name__ == "__main__":
    main()
