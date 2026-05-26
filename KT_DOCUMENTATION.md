# RAG Document Intelligence: Technical Knowledge Transfer

## 1. Project Overview

`rag-doc-intelligence` is a Python Streamlit application for asking natural-language questions over PDF documents using Retrieval-Augmented Generation.

The application lets a user:

- Upload one or more searchable PDF files.
- Validate that the PDFs contain extractable text.
- Ingest the PDFs into a local Chroma vectorstore.
- Split PDF text into overlapping chunks.
- Embed chunks using Chroma's local ONNX MiniLM embedding model.
- Retrieve relevant chunks for a user question.
- Generate a grounded RAG answer using an OpenAI-compatible chat model.
- Show citations from retrieved PDF chunks.
- Show a side-by-side plain LLM answer for comparison.

The project is designed to run locally and as a Docker-based Hugging Face Space.

Live demo declared in the README:

```text
https://huggingface.co/spaces/devz002/rag-doc-intelligence
```

No API keys or local `.env` secret values are included in this document.

---

## 2. Repository File Inventory

Meaningful project files inspected:

```text
rag-doc-intelligence/
  README.md
  requirements.txt
  Dockerfile
  .dockerignore
  .gitignore
  .env.example
  .env                  # local ignored secrets file; content intentionally not documented
  .streamlit/
    config.toml
  data/
    .gitkeep
  vectorstore/          # generated, gitignored runtime Chroma data
  src/
    __init__.py
    app.py
    ingest.py
    model_config.py
    retriever.py
    vectorstore_utils.py
```

Ignored/local/generated folders:

- `.venv/`: local virtual environment, not part of deployment.
- `.git/`: repository metadata.
- `data/*.pdf`: user-provided documents, gitignored for privacy and size.
- `vectorstore/`: generated Chroma vectorstore, gitignored.
- `.streamlit/secrets.toml`: ignored local Streamlit secrets file if created.

Current generated vectorstore behavior observed:

```text
vectorstore/
  <uuid>/
    length.bin
  <uuid>/
    length.bin
```

This indicates Chroma has generated local index artifacts. The folder is intentionally not committed and may be recreated by ingestion.

---

## 3. File-by-File Technical Summary

### `README.md`

The README contains Hugging Face Spaces metadata at the top:

```yaml
title: RAG Document Intelligence
sdk: docker
app_port: 7860
pinned: false
```

It documents:

- Project purpose.
- Stack.
- Project structure.
- Local setup.
- OpenRouter and OpenAI environment configuration.
- PDF ingestion.
- Streamlit startup.
- Hugging Face Spaces deployment.
- The fact that `vectorstore/` is generated and ignored.

Important behavior documented in README:

- Embeddings run locally using Chroma's ONNX `all-MiniLM-L6-v2` model.
- `sentence-transformers`, `scipy`, `scikit-learn`, and `torch` are intentionally avoided.
- OpenRouter defaults to `openrouter/free`.
- If OpenRouter reports `No endpoints found`, the model should be updated to a currently available free model.
- If prior installs failed because of SciPy or sentence-transformer downloads, the virtual environment should be rebuilt.

---

### `requirements.txt`

Dependencies:

```text
streamlit
python-dotenv
pypdf
chromadb
tiktoken
langchain
langchain-classic
langchain-community
langchain-openai
langchain-chroma
langchain-text-splitters
```

Purpose of each dependency:

- `streamlit`: web UI and chat interface.
- `python-dotenv`: loads `.env` during local development.
- `pypdf`: validates PDFs and extracts text diagnostics.
- `chromadb`: local vector database and ONNX MiniLM embedding backend.
- `tiktoken`: tokenizer support used by LangChain/OpenAI ecosystem.
- `langchain`: base LangChain framework.
- `langchain-classic`: provides legacy/classic chains such as `RetrievalQA` under LangChain v1.
- `langchain-community`: provides `PyPDFDirectoryLoader`.
- `langchain-openai`: provides `ChatOpenAI`.
- `langchain-chroma`: LangChain integration for Chroma.
- `langchain-text-splitters`: provides `RecursiveCharacterTextSplitter`.

Notably absent:

- `sentence-transformers`
- `scipy`
- `scikit-learn`
- `torch`

These heavy dependencies were intentionally avoided to reduce install time and deployment risk.

---

### `.env.example`

Defines the supported environment variables:

```text
EMBEDDING_MODEL=chroma-default-onnx-minilm-l6-v2

LLM_PROVIDER=openrouter

OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openrouter/free

OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

Important notes:

- `.env.example` is safe to commit.
- `.env` must not be committed.
- The app treats placeholder API keys as missing.
- Only one chat provider is selected through `LLM_PROVIDER`.
- Embeddings are local and do not require OpenAI/OpenRouter.

---

### `.gitignore`

Ignores:

```text
.env
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
vectorstore/
data/*.pdf
!data/.gitkeep
.streamlit/secrets.toml
```

Purpose:

- Prevents secrets from being committed.
- Prevents local virtual environments and Python cache files from being committed.
- Keeps private PDFs out of git.
- Keeps generated vectorstore artifacts out of git.
- Keeps `data/.gitkeep` tracked so the folder exists.

---

### `.dockerignore`

Excludes from Docker build context:

```text
.git/
.gitignore
.env
.env.*
!.env.example
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.cache/
data/
vectorstore/
*.pdf
*.sqlite
*.sqlite3
*.db
.streamlit/secrets.toml
*.log
```

Purpose:

- Prevents secrets from entering Docker images.
- Prevents local virtual environments and caches from bloating the image.
- Prevents private PDFs and generated vectorstores from being baked into the image.
- Keeps `.env.example` available as documentation.

---

### `.streamlit/config.toml`

Configuration:

```toml
[server]
port = 7860
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 100

[browser]
gatherUsageStats = false
```

Purpose:

- Uses port `7860`, required by Docker Hugging Face Spaces.
- Binds to `0.0.0.0` so the container can receive external traffic.
- Disables browser usage stats.
- Sets max upload size to 100 MB.
- Disables CORS and XSRF protection to fix Streamlit file upload `403` issues behind the Hugging Face proxy.

---

### `Dockerfile`

Uses:

```dockerfile
FROM python:3.12-slim
```

Key behavior:

- Sets environment defaults:
  - `PYTHONUNBUFFERED=1`
  - `PIP_NO_CACHE_DIR=1`
  - `MAX_PDF_UPLOAD_MB=100`
  - `PORT=7860`
  - `STREAMLIT_SERVER_MAX_UPLOAD_SIZE=100`
  - `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`
- Sets `WORKDIR /app`.
- Copies and installs `requirements.txt`.
- Copies the full project.
- Creates `data` and `vectorstore`.
- Applies `chmod -R 777 data vectorstore` so the containerized Streamlit process can write uploaded PDFs and Chroma artifacts.
- Exposes port `7860`.
- Adds a healthcheck against:

```text
http://127.0.0.1:7860/_stcore/health
```

- Starts the app with:

```text
streamlit run src/app.py --server.port=7860 --server.address=0.0.0.0
```

---

### `data/.gitkeep`

Keeps the `data/` directory in git.

Actual PDFs are ignored through `.gitignore` and `.dockerignore`.

---

### `src/__init__.py`

Empty package marker.

Allows `src` to behave as a Python package for relative imports.

---

## 4. Source Code Architecture

### `src/model_config.py`

Responsible for:

- Loading `.env`.
- Selecting the embedding model.
- Creating the local Chroma ONNX embedding adapter.
- Selecting the LLM provider.
- Creating `ChatOpenAI` clients for OpenAI or OpenRouter.

Important constants:

```text
DEFAULT_EMBEDDING_MODEL = chroma-default-onnx-minilm-l6-v2
DEFAULT_OPENAI_MODEL = gpt-4o-mini
DEFAULT_OPENROUTER_BASE_URL = https://openrouter.ai/api/v1
DEFAULT_OPENROUTER_MODEL = openrouter/free
SUPPORTED_LLM_PROVIDERS = {"openai", "openrouter"}
```

Embedding behavior:

- The app supports aliases such as:
  - `chroma-default-onnx-minilm-l6-v2`
  - `chroma/onnx-minilm-l6-v2`
  - `onnx-mini-lm-l6-v2`
  - `onnx_mini_lm_l6_v2`
  - `sentence-transformers/all-minilm-l6-v2`
- All aliases normalize to the local Chroma ONNX MiniLM backend.
- If another embedding model is configured, the app raises an error.

The class `ChromaONNXMiniLMEmbeddings` adapts Chroma's internal `ONNXMiniLM_L6_V2` embedding function to LangChain's `Embeddings` interface.

It implements:

```text
embed_documents(texts)
embed_query(text)
```

Chat provider behavior:

- `LLM_PROVIDER=openrouter`:
  - Uses `OPENROUTER_API_KEY`
  - Uses `OPENROUTER_BASE_URL`
  - Uses `OPENROUTER_MODEL`
- `LLM_PROVIDER=openai`:
  - Uses `OPENAI_API_KEY`
  - Uses `OPENAI_MODEL`
- `ChatOpenAI` is used for both providers.
- OpenRouter works because it exposes an OpenAI-compatible API.

Placeholder API keys are treated as missing. This prevents accidental attempts to call APIs with example values.

---

### `src/vectorstore_utils.py`

Responsible for Chroma/vectorstore safety and Windows/OneDrive access handling.

Defines:

```text
VectorstoreAccessError
```

Detects access problems from:

- `PermissionError`
- `OSError` errno:
  - `EACCES`
  - `EPERM`
- Windows errors:
  - `5`
  - `32`
- Error text such as:
  - `access is denied`
  - `permission denied`
  - `being used by another process`
  - `database is locked`
  - `readonly database`
  - `attempt to write a readonly database`

Important functions:

- `iter_exception_chain(exc)`: walks exception cause/context chain.
- `is_vectorstore_access_error(exc)`: detects lock/permission issues.
- `vectorstore_access_message(path, exc)`: creates a user-friendly error message.
- `ensure_vectorstore_directory(path)`: creates/validates the vectorstore directory.
- `close_chroma_client(vector_store)`: best-effort closes Chroma client handles and runs garbage collection.

The error message specifically advises:

- Close running Streamlit/Python processes.
- Retry.
- If inside OneDrive, pause sync or move the project outside OneDrive.

---

### `src/ingest.py`

Responsible for loading PDFs, validating extractable text, chunking, embedding, and persisting Chroma.

Important paths:

```text
PROJECT_ROOT = parent of src/
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
COLLECTION_NAME = "rag_doc_intelligence"
```

PDF text diagnostic behavior:

- Uses `pypdf.PdfReader`.
- Detects:
  - parse errors
  - encrypted/password-protected PDFs
  - zero-page PDFs
  - PDFs with no extractable/selectable text
- Provides OCR guidance for scanned/image-only PDFs:

```text
Scanned/image-only PDFs need OCR preprocessing first.
Create a searchable PDF with OCR, then upload that OCR-processed file.
```

Main ingestion function:

```text
ingest_pdfs(data_dir=DATA_DIR, persist_directory=VECTORSTORE_DIR, reset=True)
```

Step-by-step:

1. Ensure `data/` exists.
2. Find all `*.pdf` files.
3. If no PDFs exist, raise `FileNotFoundError`.
4. Inspect each PDF using `pypdf`.
5. Block ingestion for unreadable, encrypted, or zero-page PDFs.
6. If no PDFs contain extractable text, raise a detailed OCR-oriented error.
7. Load pages with `PyPDFDirectoryLoader`.
8. Drop empty page documents.
9. Split text with:

```text
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

10. Create local Chroma ONNX embeddings using `get_embeddings()`.
11. Ensure `vectorstore/` exists.
12. Create a Chroma collection:

```text
collection_name = rag_doc_intelligence
persist_directory = vectorstore/
```

13. If `reset=True`, call `reset_collection()`.
14. Add chunks with `add_documents(chunks)`.
15. Persist if the Chroma object exposes `persist()`.
16. Close Chroma client handles in `finally`.
17. Return stats:
    - number of PDFs
    - loaded page documents
    - text-containing page documents
    - chunks
    - warnings

CLI behavior:

```powershell
python .\src\ingest.py
```

Optional CLI arguments:

```text
--data-dir
--persist-directory
--no-reset
```

By default, ingestion rebuilds the collection. With `--no-reset`, it appends to the existing collection.

---

### `src/retriever.py`

Responsible for loading the vectorstore, creating the retriever, creating the RAG chain, creating the plain LLM, and formatting citations.

Important compatibility detail:

```python
from langchain_classic.chains import RetrievalQA
```

This avoids LangChain v1 compatibility issues where older examples importing from `langchain.chains` no longer work reliably.

Vectorstore behavior:

- `vectorstore_exists()` returns true if `vectorstore/` exists and contains files.
- Access errors are converted into `VectorstoreAccessError`.
- `get_vectorstore()` raises a helpful error if the vectorstore does not exist.

Retriever behavior:

```text
search_type = "mmr"
search_kwargs = {"k": 5, "fetch_k": 20}
```

This means:

- Fetch up to 20 candidates.
- Return 5 diversified/relevant chunks using Maximal Marginal Relevance.

RAG chain behavior:

```text
RetrievalQA.from_chain_type(
  llm=get_chat_model(temperature=0),
  chain_type="stuff",
  retriever=retriever,
  return_source_documents=True
)
```

Meaning:

- Uses deterministic temperature `0`.
- Uses LangChain's `stuff` chain type.
- Injects retrieved chunks into the prompt context.
- Returns source documents for citations.

Plain LLM behavior:

- Uses the same chat provider/model.
- Receives only the raw user question.
- Does not use retrieved document context.

Citation formatting:

- Extracts file name from document metadata `source`.
- Converts zero-indexed page metadata to user-facing page number by adding 1.
- Produces labels like:

```text
[1] document.pdf, page 3
```

- Excerpts are normalized whitespace and truncated to 500 characters.

---

### `src/app.py`

Streamlit UI and runtime orchestration.

Main UI behavior:

- Page title: `RAG Document Intelligence`.
- Layout: wide.
- Sidebar:
  - Shows document upload controls.
  - Shows embedding model.
  - Shows selected LLM provider/model.
  - Shows whether the required API key is detected.
  - Shows PDF count in `data/`.
  - Allows multiple PDF uploads.
  - Has `Save and ingest PDFs`.
  - Shows vectorstore readiness.
  - Allows clearing chat history.
- Main panel:
  - Displays chat history.
  - Disables chat until both:
    - chat provider API key is configured
    - vectorstore exists
  - Shows RAG answer and plain LLM answer side by side.

Upload configuration:

```text
DEFAULT_MAX_PDF_UPLOAD_MB = 100
MAX_PDF_UPLOAD_MB from environment
STREAMLIT_SERVER_MAX_UPLOAD_SIZE from environment
```

Upload validation:

1. File must end with `.pdf`.
2. File must not be empty.
3. File must be below `MAX_PDF_UPLOAD_MB`.
4. File header must start with `%PDF-`.
5. `pypdf` must be able to inspect text.
6. Scanned/image-only PDFs are warned and skipped.

Filename safety:

- Uploaded filenames are sanitized with a regex.
- Unsafe characters are replaced with `_`.
- If a filename already exists, the app appends `_1`, `_2`, etc.

Chroma cache/resource handling:

- `cached_rag_chain()` and `cached_plain_llm()` use Streamlit `@st.cache_resource`.
- Before rebuilding the vectorstore, the app calls `release_cached_chroma_resources()`.
- That function:
  - finds the cached chain's vectorstore
  - closes the Chroma client
  - clears Streamlit resource cache
  - runs garbage collection

This was added to reduce locked SQLite/vectorstore issues on Windows and OneDrive.

Query behavior:

```text
run_comparison(question)
```

Per question:

1. Invoke cached RAG chain with `{"query": question}`.
2. Invoke plain LLM with the raw question.
3. Return:
   - original question
   - RAG answer
   - plain LLM answer
   - formatted sources
4. Render two columns:
   - RAG answer with source expanders
   - plain LLM answer with note that it does not use the vectorstore

OpenRouter error handling:

If an exception contains:

```text
no endpoints found
```

or OpenRouter-related `404`, the app shows a hint to update `OPENROUTER_MODEL` to a currently listed free model.

---

## 5. End-to-End Architecture

High-level architecture:

```text
User
  -> Streamlit UI
    -> PDF upload/save
      -> data/*.pdf
        -> pypdf validation
        -> PyPDFDirectoryLoader
        -> RecursiveCharacterTextSplitter
        -> Chroma ONNX MiniLM embeddings
        -> Chroma vectorstore persisted in vectorstore/
          -> Retriever using MMR
            -> RetrievalQA RAG chain
              -> ChatOpenAI
                -> OpenRouter or OpenAI
                  -> RAG answer + source documents
    -> Plain ChatOpenAI call
      -> Plain answer without document context
```

Main components:

- UI: Streamlit.
- PDF parsing: `pypdf` plus LangChain `PyPDFDirectoryLoader`.
- Chunking: LangChain text splitter.
- Embeddings: local Chroma ONNX MiniLM.
- Vector DB: ChromaDB persisted to `vectorstore/`.
- Retrieval: Chroma retriever using MMR.
- RAG chain: LangChain Classic `RetrievalQA`.
- LLM provider: OpenAI-compatible `ChatOpenAI`.
- Deployment: Docker on Hugging Face Spaces.

---

## 6. Step-by-Step Execution Flow

### 6.1 Local Setup

PowerShell:

```powershell
cd "C:\Users\DEVIKA\OneDrive - UNSW\Term 1 2026\rag-doc-intelligence"

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env
```

Edit `.env` for OpenRouter:

```text
EMBEDDING_MODEL=chroma-default-onnx-minilm-l6-v2
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=<your-openrouter-key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openrouter/free
```

Or edit `.env` for OpenAI:

```text
EMBEDDING_MODEL=chroma-default-onnx-minilm-l6-v2
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-openai-key>
OPENAI_MODEL=gpt-4o-mini
```

Do not commit `.env`.

---

### 6.2 Rebuild Virtual Environment After Failed Heavy Installs

If a previous install failed while downloading SciPy, sentence-transformers, torch, or similar packages:

```powershell
cd "C:\Users\DEVIKA\OneDrive - UNSW\Term 1 2026\rag-doc-intelligence"

Get-Process python,pip -ErrorAction SilentlyContinue | Stop-Process -Force
Remove-Item -Recurse -Force .\.venv -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:TEMP\pip-*" -ErrorAction SilentlyContinue

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

The current dependency set avoids `sentence-transformers` and SciPy.

---

### 6.3 Ingest Flow from CLI

1. Place searchable PDFs in:

```text
data/
```

2. Run:

```powershell
python .\src\ingest.py
```

3. The ingestion script:
   - scans `data/*.pdf`
   - validates PDFs with `pypdf`
   - loads pages with `PyPDFDirectoryLoader`
   - drops empty pages
   - splits text into 1000-character chunks with 200-character overlap
   - embeds chunks locally with Chroma ONNX MiniLM
   - stores them in Chroma under collection `rag_doc_intelligence`
   - writes generated artifacts to `vectorstore/`

4. Expected output shape:

```text
Ingested <n> PDF(s), loaded <n> page document(s) (<n> with extractable text), and stored <n> chunk(s).
Embedding model: chroma-default-onnx-minilm-l6-v2 (local Chroma ONNX MiniLM)
```

---

### 6.4 Rebuild Vectorstore

If the embedding backend changes, documents change, or the store is stale:

```powershell
Get-Process python,streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
Remove-Item -Recurse -Force .\vectorstore -ErrorAction SilentlyContinue
python .\src\ingest.py
```

If Windows or OneDrive reports access denied, do not repeatedly delete the locked folder. Instead:

1. Close Streamlit/browser sessions using the app.
2. Stop Python processes.
3. Pause OneDrive sync or move the repo outside OneDrive.
4. Re-run ingestion.

The app itself rebuilds by calling Chroma `reset_collection()` rather than relying only on directory deletion.

---

### 6.5 App Startup Flow

Run:

```powershell
streamlit run .\src\app.py
```

Startup sequence:

1. Streamlit imports `app.py`.
2. Page config is set.
3. `.env` is loaded through `model_config.py`.
4. The sidebar renders.
5. The app checks selected provider:
   - OpenRouter or OpenAI
6. The app checks for a non-placeholder API key.
7. Runtime directories are created:
   - `data/`
   - `vectorstore/`
8. Existing PDFs are counted.
9. Vectorstore readiness is checked.
10. Chat input is disabled until:
    - API key is configured
    - vectorstore exists

Because `.streamlit/config.toml` sets port `7860`, local runs may also use `7860` unless overridden.

---

### 6.6 User Upload Flow

When the user uploads PDFs in the sidebar and clicks `Save and ingest PDFs`:

1. Each uploaded file is validated.
2. Invalid files are rejected:
   - not `.pdf`
   - empty
   - above size limit
   - missing `%PDF-` header
   - unreadable PDF
   - encrypted/password-protected
3. Scanned/image-only PDFs produce OCR guidance.
4. Valid files are saved into `data/`.
5. Filenames are sanitized.
6. Duplicate filenames get numeric suffixes.
7. Cached Chroma resources are released.
8. `ingest_pdfs()` runs.
9. Cached resources are released again.
10. The UI reports:
    - number of PDFs
    - number of chunks
    - warnings, if any
    - saved filenames

---

### 6.7 Query and RAG Response Flow

When the user asks a question:

1. The question appears as a user chat message.
2. `run_comparison(question)` is called.
3. RAG path:
   - cached RetrievalQA chain is loaded
   - retriever queries Chroma using MMR
   - top 5 source chunks are returned
   - chunks are stuffed into the RAG prompt
   - selected chat model generates a grounded answer
4. Plain LLM path:
   - the same chat provider/model receives only the raw question
   - no PDF context is supplied
5. The UI renders:
   - left column: RAG answer
   - source expanders with page-aware excerpts
   - right column: plain LLM answer
6. The full comparison record is stored in Streamlit session state.

---

### 6.8 Docker and Hugging Face Spaces Deployment Flow

The project is configured for Docker Spaces.

Local Docker build:

```powershell
docker build -t rag-doc-intelligence .
```

Run locally with environment file:

```powershell
docker run --env-file .env -p 7860:7860 rag-doc-intelligence
```

Hugging Face Space requirements:

- Space SDK must be Docker.
- App port must be `7860`.
- Required secrets/variables must be configured individually.
- `Dockerfile` installs Python dependencies and starts Streamlit.
- Uploaded PDFs and vectorstore are created at runtime.

Typical remote setup:

```powershell
git remote add space https://huggingface.co/spaces/devz002/rag-doc-intelligence
```

If the `space` remote was accidentally pointed to GitHub, fix it:

```powershell
git remote set-url space https://huggingface.co/spaces/devz002/rag-doc-intelligence
```

Push to GitHub:

```powershell
git push origin main
```

Push to Hugging Face Space:

```powershell
git push space main
```

If Hugging Face has template history and the project intentionally replaces it, a non-fast-forward push may occur. After confirming replacement is intended:

```powershell
git push --force-with-lease space main
```

Use `--force-with-lease`, not blind `--force`, because it is safer if the remote changed unexpectedly.

After changing `Dockerfile`, `requirements.txt`, or build-level configuration, trigger a Hugging Face Space Factory Rebuild from the Space settings. A normal restart may not fully rebuild dependencies.

---

## 7. Hugging Face Secrets and Variables

Add secrets individually in Hugging Face Space settings.

Do not paste multiple environment lines into one secret value.

Recommended OpenRouter Space configuration:

```text
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=<secret>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openrouter/free
EMBEDDING_MODEL=chroma-default-onnx-minilm-l6-v2
```

Recommended OpenAI Space configuration:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=<secret>
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=chroma-default-onnx-minilm-l6-v2
```

Hugging Face CLI, if available in the installed CLI version:

```powershell
hf auth login
hf space secrets list devz002/rag-doc-intelligence
```

If the CLI does not support secret listing in the local version, use:

```text
Hugging Face Space -> Settings -> Variables and secrets
```

Secret values should not be printed into logs or committed to git.

---

## 8. Operational Commands

### Local Run

```powershell
cd "C:\Users\DEVIKA\OneDrive - UNSW\Term 1 2026\rag-doc-intelligence"
.\.venv\Scripts\Activate.ps1
streamlit run .\src\app.py
```

### CLI Ingest

```powershell
python .\src\ingest.py
```

### Ingest with Custom Paths

```powershell
python .\src\ingest.py --data-dir .\data --persist-directory .\vectorstore
```

### Append Instead of Reset

```powershell
python .\src\ingest.py --no-reset
```

### Rebuild Vectorstore

```powershell
Get-Process python,streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
Remove-Item -Recurse -Force .\vectorstore -ErrorAction SilentlyContinue
python .\src\ingest.py
```

### Docker Build and Run

```powershell
docker build -t rag-doc-intelligence .
docker run --env-file .env -p 7860:7860 rag-doc-intelligence
```

### GitHub Push

```powershell
git status
git add README.md requirements.txt Dockerfile .dockerignore .gitignore .env.example .streamlit/config.toml src data/.gitkeep
git commit -m "Prepare RAG document intelligence app for deployment"
git push origin main
```

### Hugging Face Push

```powershell
git remote set-url space https://huggingface.co/spaces/devz002/rag-doc-intelligence
git push space main
```

### Replace Hugging Face Template History, If Intended

```powershell
git push --force-with-lease space main
```

### Hugging Face Factory Rebuild

Use the Hugging Face UI:

```text
Space -> Settings -> Factory rebuild
```

Use this after dependency, Dockerfile, or build environment changes.

---

## 9. Bottlenecks Encountered and Resolutions

### 9.1 LangChain v1 Import Compatibility

Issue:

Older LangChain examples use:

```python
from langchain.chains import RetrievalQA
```

With LangChain v1, this can fail because classic chains moved.

Resolution:

The project uses:

```python
from langchain_classic.chains import RetrievalQA
```

And includes the split LangChain packages:

```text
langchain-classic
langchain-community
langchain-openai
langchain-chroma
langchain-text-splitters
```

---

### 9.2 OpenAI Quota 429

Issue:

OpenAI requests can fail with quota/rate errors such as HTTP `429`.

Resolution:

The app supports OpenRouter as an OpenAI-compatible provider.

Set:

```text
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=<key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openrouter/free
```

---

### 9.3 OpenRouter Model 404 or No Endpoints

Issue:

Specific free OpenRouter models may disappear, become unavailable, or return `No endpoints found`.

Resolution:

The default model was changed to:

```text
OPENROUTER_MODEL=openrouter/free
```

This routes to available free models rather than pinning to a single disappearing endpoint.

The Streamlit app also detects OpenRouter 404/no-endpoint messages and tells the user to update `OPENROUTER_MODEL` from the current free model list.

---

### 9.4 `sentence-transformers` / SciPy Install Timeout

Issue:

Installing `sentence-transformers` can pull heavy packages such as SciPy, scikit-learn, and torch. On Windows or constrained environments this can time out or fail.

Resolution:

The project removed that dependency path and uses Chroma's local ONNX MiniLM embedding function directly.

Benefits:

- Smaller install.
- No OpenAI embedding cost.
- Avoids SciPy/torch install failures.
- Works in local and Docker deployment.

---

### 9.5 Windows/OneDrive Vectorstore Access Denied

Issue:

Chroma persists data using local files. On Windows, especially under OneDrive, SQLite/index files can be locked by:

- running Streamlit processes
- Python processes
- OneDrive sync
- stale Chroma client handles

This can cause:

```text
Access is denied
database is locked
being used by another process
readonly database
```

Resolution in code:

- `vectorstore_utils.py` detects these errors.
- The app closes cached Chroma clients before rebuilding.
- Streamlit cache is cleared.
- Garbage collection is triggered.
- Ingestion closes Chroma clients in `finally`.
- Rebuild uses `reset_collection()` instead of relying only on folder deletion.

Operational resolution:

- Stop Python/Streamlit processes.
- Pause OneDrive sync.
- Move project outside OneDrive if needed.
- Avoid repeated forced deletion while files are locked.

---

### 9.6 Hugging Face Docker SDK and Port

Issue:

Hugging Face Docker Spaces require the app to listen on the configured port, typically `7860`. Streamlit defaults to `8501` locally if not configured.

Resolution:

The project sets port `7860` in three places:

- README Hugging Face metadata: `app_port: 7860`
- `.streamlit/config.toml`: `port = 7860`
- `Dockerfile`: `--server.port=7860`

The Docker command also binds to:

```text
0.0.0.0:7860
```

---

### 9.7 Misconfigured `space` Remote

Issue:

The `space` git remote can accidentally point to GitHub instead of Hugging Face.

Resolution:

Set the remote explicitly:

```powershell
git remote set-url space https://huggingface.co/spaces/devz002/rag-doc-intelligence
```

Then push:

```powershell
git push space main
```

---

### 9.8 Non-Fast-Forward Push to Hugging Face

Issue:

A newly created Hugging Face Space may contain template commits. Pushing a replacement project can fail as non-fast-forward.

Resolution:

If the intent is to replace the Space template history:

```powershell
git push --force-with-lease space main
```

This should only be done after confirming the remote history is disposable.

---

### 9.9 Streamlit Upload 403 on Hugging Face

Issue:

File uploads on Hugging Face can return `403` because of Streamlit CORS/XSRF behavior behind the Space proxy.

Resolution:

`.streamlit/config.toml` sets:

```toml
enableCORS = false
enableXsrfProtection = false
```

This allows uploads to work behind the Hugging Face reverse proxy.

---

### 9.10 No Text Chunks from Scanned PDFs

Issue:

Scanned/image-only PDFs do not contain selectable text, so `pypdf` and `PyPDFDirectoryLoader` cannot extract meaningful content.

Resolution:

The app now performs pypdf diagnostics before ingestion and upload save.

It reports:

- parse errors
- encrypted PDFs
- zero-page PDFs
- no selectable text
- OCR guidance

Required fix for users:

```text
Run OCR first, create a searchable PDF, then upload the OCR-processed file.
```

OCR is intentionally not bundled to avoid heavy Docker dependencies.

---

### 9.11 Hugging Face Secret Misconfiguration

Issue:

Pasting multiple environment variables into one Hugging Face secret results in the app not seeing the expected individual variables.

Resolution:

Create each variable/secret individually:

```text
LLM_PROVIDER
OPENROUTER_API_KEY
OPENROUTER_BASE_URL
OPENROUTER_MODEL
OPENAI_API_KEY
OPENAI_MODEL
EMBEDDING_MODEL
```

Only configure the provider-specific API key that is needed.

---

## 10. Security and Privacy Notes

- `.env` is ignored and must never be committed.
- `.dockerignore` excludes `.env` and `.env.*` except `.env.example`.
- `.streamlit/secrets.toml` is ignored.
- PDF files are ignored because they may contain private documents.
- `vectorstore/` is ignored because embeddings/indexes are generated from private PDFs.
- Hugging Face secrets should be stored as individual Space secrets.
- Public Hugging Face Spaces have shared runtime state; uploaded PDFs may be visible to the running app context and should not be treated as private multi-user storage.
- API keys are used server-side through environment variables, not displayed in the UI.

---

## 11. Limitations

Current limitations:

- No OCR pipeline is included.
- Scanned/image-only PDFs must be OCR-processed externally.
- Hugging Face free Spaces have ephemeral runtime storage unless persistent storage is configured.
- Uploaded PDFs and vectorstore may disappear after a Space restart/rebuild.
- Free OpenRouter model availability can change.
- OpenRouter free models may have rate limits, latency, or quality variance.
- Upload size is capped at 100 MB by app/container config.
- PDF extraction quality depends on `pypdf`; complex layouts, tables, columns, and footnotes may be imperfect.
- The vectorstore is global for the app runtime, not per user/session.
- Reingestion with reset rebuilds the shared collection from all PDFs in `data/`.
- Only one embedding backend is currently supported.
- There are no dedicated automated tests or CI files in the inspected project.
- Secrets management is external to the repo and must be configured correctly in local `.env` or Hugging Face settings.

---

## 12. Future Improvements

Recommended improvements:

- Add OCR support as an optional pipeline using tools such as OCRmyPDF or Tesseract.
- Add persistent Hugging Face storage or external object storage for uploaded PDFs and vectorstores.
- Add per-session or per-user document isolation.
- Add delete/manage-document controls in the UI.
- Add tests for:
  - PDF validation
  - filename sanitization
  - provider config
  - vectorstore access error detection
  - source formatting
- Pin dependency versions for more reproducible Docker builds.
- Add structured logging.
- Add chunking controls in the UI.
- Add support for additional embedding providers.
- Add fallback model selection for OpenRouter when `openrouter/free` is unavailable.
- Add document metadata display and ingestion status history.
- Add evaluation examples comparing RAG vs plain LLM answer quality.
- Add authentication if deployed with sensitive documents.

---

## 13. Summary for Claude

This project is a Docker-deployable Streamlit RAG application for PDF question answering. It uses local Chroma ONNX MiniLM embeddings, ChromaDB persistence, LangChain Classic RetrievalQA, and OpenAI-compatible chat completions through either OpenRouter or OpenAI. The main user flow is upload PDFs, validate text extraction, ingest into Chroma, ask questions, view a source-grounded RAG answer with citations, and compare it against a plain LLM answer.

The project was hardened for real deployment issues: LangChain v1 import changes, OpenAI quota errors, OpenRouter free model availability, heavyweight embedding dependency failures, Windows/OneDrive file locking, Hugging Face Docker port requirements, misconfigured git remotes, non-fast-forward Space history, Streamlit upload 403s, scanned PDF failures, and Hugging Face secret misconfiguration.
