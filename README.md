---
title: RAG Document Intelligence
sdk: docker
app_port: 7860
pinned: false
---

# RAG Document Intelligence

Live demo: https://huggingface.co/spaces/devz002/rag-doc-intelligence

A Streamlit chat app for asking questions about PDF documents. It builds a ChromaDB vectorstore from PDFs using local Chroma ONNX MiniLM embeddings, answers with LangChain RetrievalQA, cites retrieved source chunks, and shows a side-by-side comparison against a plain LLM answer.

## Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Local Chroma ONNX MiniLM embeddings
- OpenAI or OpenRouter chat completions
- Hugging Face Spaces

## Project Structure

```text
rag-doc-intelligence/
  data/              # Local PDFs; PDF files are gitignored
  vectorstore/       # Generated ChromaDB index; gitignored
  src/
    app.py           # Streamlit UI
    ingest.py        # PDF ingestion and embedding
    retriever.py     # Chroma retriever and QA helpers
  .env.example
  .dockerignore
  .gitignore
  Dockerfile
  requirements.txt
```

## Local Setup

```powershell
cd rag-doc-intelligence
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Embeddings run locally with Chroma's default ONNX `all-MiniLM-L6-v2` model by default. This keeps embeddings free and avoids installing `sentence-transformers`, `scipy`, `scikit-learn`, and `torch`. The first ingest may download the ONNX model into your Chroma cache.

Chat completions can use OpenRouter or OpenAI.

For OpenRouter, edit `.env` with:

```text
EMBEDDING_MODEL=chroma-default-onnx-minilm-l6-v2
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-real-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openrouter/free
```

`openrouter/free` uses OpenRouter's free-model router, which avoids pinning the app to one free endpoint that may later disappear. To pin a specific model instead, choose a current free model ID from <https://openrouter.ai/models?max_price=0>. If Streamlit reports `No endpoints found`, update `OPENROUTER_MODEL` to a currently listed model, save `.env`, and restart Streamlit.

For OpenAI, edit `.env` with:

```text
EMBEDDING_MODEL=chroma-default-onnx-minilm-l6-v2
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-4o-mini
```

Do not commit `.env`. It is ignored by `.gitignore`.

PowerShell example for OpenRouter:

```powershell
Copy-Item .env.example .env
notepad .env
pip install -r requirements.txt
```

If a previous install failed while downloading SciPy or `sentence-transformers`, close running Python/Streamlit processes and rebuild the virtual environment:

```powershell
cd "C:\Users\DEVIKA\OneDrive - UNSW\Term 1 2026\rag-doc-intelligence"
Get-Process python,pip -ErrorAction SilentlyContinue | Stop-Process -Force
Remove-Item -Recurse -Force .\.venv -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:TEMP\pip-*" -ErrorAction SilentlyContinue
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

## Ingest PDFs

Place PDF files in `data/`, then build the Chroma vectorstore:

```powershell
python src/ingest.py
```

The app also supports uploading PDFs from the Streamlit sidebar and ingesting them there.

If you previously ingested PDFs with OpenAI embeddings, sentence-transformer embeddings, or any other embedding backend, rebuild the vectorstore so Chroma uses the same embedding backend for ingestion and retrieval:

```powershell
Remove-Item -Recurse -Force .\vectorstore -ErrorAction SilentlyContinue
python .\src\ingest.py
```

## Run the App

```powershell
streamlit run src/app.py
```

Open the local Streamlit URL, upload or ingest PDFs, then ask a question. The response view shows:

- A RAG answer grounded in retrieved PDF chunks
- Source citations with page-aware excerpts
- A plain LLM answer for comparison
- The selected chat provider/model and local embedding model in the UI

## Hugging Face Spaces Deployment

1. Create a new Space with the Docker SDK.
2. Upload this project or push it to the Space repository.
3. Add either `OPENROUTER_API_KEY` or `OPENAI_API_KEY` as a Space secret.
4. Add `LLM_PROVIDER` as `openrouter` or `openai`.
5. The `Dockerfile` installs `requirements.txt` and starts Streamlit on `0.0.0.0:7860`.
6. Use the sidebar to upload PDFs and build the runtime vectorstore.

`vectorstore/` is generated and intentionally ignored. If the Space restarts, rebuild the vectorstore from uploaded or bundled PDFs.
