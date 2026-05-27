# Resume Bullets — RAG Document Intelligence System

---

## Primary bullet (use this as the project headline)

**RAG Document Intelligence System** | LangChain · ChromaDB · Streamlit · Docker · Hugging Face Spaces

- Built and deployed a full-stack Retrieval-Augmented Generation (RAG) application enabling
  natural-language Q&A over user-uploaded PDF documents, with source-grounded answers and
  per-chunk citations — live at huggingface.co/spaces/devz002/rag-doc-intelligence

---

## Supporting bullets (pick 3–4 that fit the role you're applying for)

### Architecture & ML Engineering
- Designed a two-phase RAG pipeline: an offline ingestion stage (PDF loading → chunking →
  local ONNX MiniLM embedding → ChromaDB persistence) and an online query stage (MMR
  retrieval of top-5 from 20 candidates → LangChain RetrievalQA chain → cited answer)
- Implemented Maximal Marginal Relevance (MMR) retrieval over ChromaDB with fetch_k=20,
  k=5 to balance relevance and chunk diversity, reducing repeated context in LLM prompts
- Used Chroma's local ONNX MiniLM-L6-v2 embedding model to eliminate OpenAI embedding
  API costs and remove heavy dependencies (PyTorch, SciPy, sentence-transformers),
  cutting Docker image size and install time significantly
- Built a side-by-side RAG vs. plain LLM comparison view demonstrating hallucination
  reduction — grounded answers include source file and page-level citations; plain LLM
  answers are shown without document context for direct comparison

### Data Engineering & Production Hardening
- Engineered pre-ingestion PDF validation using pypdf to detect and block scanned/image-only
  PDFs, encrypted files, zero-page PDFs, and parse errors before they silently corrupt the
  vectorstore — with user-facing OCR guidance for unsupported files
- Built a vectorstore safety layer (vectorstore_utils.py) to detect and recover from
  SQLite file-locking errors on Windows/OneDrive environments, including graceful Chroma
  client teardown, Streamlit cache clearing, and garbage collection before rebuilds
- Implemented upload security controls: PDF magic-byte header validation (%PDF-),
  file size capping at 100 MB, regex-based filename sanitisation, and collision-safe
  deduplication (appending _1, _2 suffixes) to prevent path conflicts

### Deployment & DevOps
- Containerised the application with Docker (python:3.12-slim), configuring Streamlit on
  port 7860 across three layers (README metadata, config.toml, Dockerfile CMD) for
  correct Hugging Face Spaces Docker SDK deployment
- Configured CORS and XSRF protection settings in Streamlit to resolve file upload 403
  errors behind the Hugging Face reverse proxy — identified and fixed a non-obvious
  production deployment issue
- Designed a dual LLM provider architecture (OpenAI and OpenRouter) via a unified
  model_config.py abstraction, enabling cost-free development using OpenRouter's free
  tier and production use with OpenAI gpt-4o-mini, switchable via a single environment
  variable

### Security & Best Practices
- Applied production secrets hygiene: .env excluded via .gitignore and .dockerignore,
  .env.example committed as documentation, placeholder API keys treated as missing at
  runtime — preventing accidental API calls with example credentials
- Structured the project as a proper Python package (src/ with __init__.py), with
  @st.cache_resource for chain/model caching, CLI flags for ingestion (--no-reset for
  append vs. rebuild), and environment-driven configuration throughout

---

## One-liner version (for space-constrained resumes)

- Built and deployed a Docker-containerised RAG application (LangChain, ChromaDB,
  Streamlit) on Hugging Face Spaces — featuring local ONNX embeddings, MMR retrieval,
  PDF validation, per-chunk citations, and a side-by-side RAG vs. plain LLM hallucination
  comparison; hardened for production with secrets management, file-locking recovery,
  and upload security controls

---

## Notes on tailoring

| Role you're applying for       | Lead with                                      |
|-------------------------------|------------------------------------------------|
| Data Scientist                 | RAG pipeline design, MMR retrieval, hallucination comparison |
| ML Engineer                    | Architecture, embedding choice, LangChain chain design       |
| Data / AI Engineer             | Ingestion pipeline, vectorstore safety, Docker deployment    |
| Full-stack / Software Engineer | Streamlit UI, Docker, dual-provider abstraction, security    |

