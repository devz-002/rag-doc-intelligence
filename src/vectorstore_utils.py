from __future__ import annotations

import errno
import gc
from pathlib import Path
from typing import Any, Iterator


ACCESS_ERROR_ERRNOS = {errno.EACCES, errno.EPERM}
ACCESS_ERROR_WINERRORS = {5, 32}
ACCESS_ERROR_TEXT = (
    "access is denied",
    "permission denied",
    "being used by another process",
    "database is locked",
    "readonly database",
    "attempt to write a readonly database",
)


class VectorstoreAccessError(RuntimeError):
    """Raised when Chroma's persisted vectorstore cannot be accessed safely."""


def iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_vectorstore_access_error(exc: BaseException) -> bool:
    for chained_exc in iter_exception_chain(exc):
        if isinstance(chained_exc, PermissionError):
            return True
        if isinstance(chained_exc, OSError):
            if getattr(chained_exc, "winerror", None) in ACCESS_ERROR_WINERRORS:
                return True
            if getattr(chained_exc, "errno", None) in ACCESS_ERROR_ERRNOS:
                return True
        message = str(chained_exc).lower()
        if any(text in message for text in ACCESS_ERROR_TEXT):
            return True
    return False


def vectorstore_access_message(persist_path: Path | str, exc: BaseException) -> str:
    return (
        f"Vectorstore at {Path(persist_path)} is locked or inaccessible: {exc}. "
        "Close any running Streamlit/Python processes that may be using it, then try again. "
        "If this project is inside OneDrive and the problem persists, pause OneDrive syncing "
        "or move the project to a local folder outside OneDrive."
    )


def ensure_vectorstore_directory(persist_path: Path | str) -> Path:
    path = Path(persist_path)
    try:
        if path.exists() and not path.is_dir():
            raise VectorstoreAccessError(
                f"Vectorstore path exists but is not a directory: {path}. "
                "Remove or rename that file, then ingest PDFs again."
            )
        path.mkdir(parents=True, exist_ok=True)
    except VectorstoreAccessError:
        raise
    except OSError as exc:
        if is_vectorstore_access_error(exc):
            raise VectorstoreAccessError(vectorstore_access_message(path, exc)) from exc
        raise
    return path


def close_chroma_client(vector_store: Any) -> None:
    """Best-effort release of Chroma's SQLite handles before rebuilding."""
    try:
        client = getattr(vector_store, "_client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()
    except Exception:
        pass
    finally:
        gc.collect()
