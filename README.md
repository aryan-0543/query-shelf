# Local PDF RAG

A small, educational Retrieval-Augmented Generation (RAG) application for
asking questions about a local PDF. The project is being built incrementally
so that PDF extraction, chunking, local embeddings, vector retrieval, prompt
construction, and Gemini answer generation can each be understood and tested
independently.

## Current status

The document-loading and page-level text-extraction components are implemented
and covered by focused tests. The remaining RAG pipeline components are still
planned and have not been implemented or verified.

The planned initial implementation will support:

- One local PDF at a time.
- Page-aware text extraction with PyMuPDF.
- Custom Python chunking.
- Local embeddings using `all-MiniLM-L6-v2`.
- Local ChromaDB storage and similarity retrieval.
- Grounded answer generation through the official Google Gen AI Python SDK.
- A command-line interface with an interactive question loop first.
- Source/page references in answers where retrieval metadata is available.

One-shot question mode is planned for a later milestone. The proposed
`app/` and `tests/` layout is an initial structure, not a permanent contract.
As the project grows, new modules or folders may be proposed when they solve a
real need; related architecture and implementation documentation should be
updated before those changes are implemented.

## Objectives

- Understand each RAG component rather than hiding it behind a framework.
- Keep the first version simple, modular, and independently testable.
- Avoid LangChain and LangGraph in the initial implementation.
- Keep Gemini-specific integration isolated so the provider can be replaced
  later.
- Keep PDF processing, embeddings, and retrieval local wherever practical.

## Technology stack

| Area | Initial technology |
| --- | --- |
| Language | Python |
| PDF extraction | PyMuPDF |
| Chunking | Custom Python implementation |
| Embeddings | `sentence-transformers` with `all-MiniLM-L6-v2` |
| Vector database | ChromaDB with local persistence |
| Answer generation | Google Gemini API |
| Gemini SDK | Official `google-genai` Python SDK |
| Configuration | `python-dotenv` and environment variables |
| Interface | Command line |
| Testing | pytest |

The embedding model runs locally. Gemini is used for answer generation only:
the initial design sends the user question and retrieved context, not the
complete PDF, to the API.

## Prerequisites

- Python compatible with the selected dependency versions. The supported
  version will be confirmed during project setup and recorded here.
- Internet access to install Python packages and download the embedding model
  on first use.
- A Google AI Studio/Gemini API key with access to a supported model.
- Sufficient local memory and disk space for the embedding model and local
  ChromaDB data.

Gemini availability, model compatibility, quotas, and rate limits are
model- and account-dependent. They must be verified with the user's API key
before a default model is finalized. Free-tier access does not mean unlimited
usage.

## Setup

From the repository root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local environment file from the template:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder value with your own Gemini API key.
Choose a Gemini model available to your account for `GEMINI_MODEL`. Do not
paste a real key into this README, `.env.example`, source code, tests, or any
committed file.

The local `.env` file is excluded by `.gitignore`. Never commit it or include
its contents in an issue, log, screenshot, or pull request.

## Configuration

The initial configuration uses:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=your_supported_gemini_model
```

`GEMINI_API_KEY` is required for answer generation. The application will
eventually fail with a clear error if it is missing or invalid, without
printing the key. `GEMINI_MODEL` keeps the model configurable rather than
hardcoding a provider model throughout the retrieval pipeline.

The proposed local ChromaDB persistence directory is `data/chroma/`. Generated
data should remain local and uncommitted.

## Usage

The full application usage will be documented here after the CLI milestone is
implemented and verified. The current manually verifiable document workflow
is:

1. Set `PDF_PATH` in `scripts/manual_load_pdf.py` to a local PDF.
2. Run `python -m scripts.manual_load_pdf` from the repository root.
3. The script reports loading progress, displays the page count, extracts text
   page by page, and labels empty pages.

The script does not execute chunking or later RAG stages. The interactive
question loop and one-shot question mode are planned for later milestones and
are not yet implemented.

## Testing

After the test suite is implemented, run:

```bash
python -m pytest
```

Unit tests should mock Gemini responses wherever practical. This keeps tests
fast, avoids unnecessary quota consumption, and prevents ordinary tests from
depending on network availability. Any live Gemini check should be explicit,
separate from the default unit suite, and quota-conscious.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) describes component boundaries and data
  flow.
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) tracks sequential
  milestones and measurable completion criteria.

## Future scope

These are future goals, not initial features:

- Multiple PDF uploads.
- Persistent document collections and metadata filtering.
- Better retrieval, hybrid search, and reranking.
- Tables, scanned PDFs, images, and charts.
- FastAPI backend.
- Authentication and a frontend.
- Docker and cloud deployment.

Each future capability should be introduced through a documented architecture
proposal when it creates new modules, folders, or configuration needs. The
project should avoid adding abstractions before they are justified by an
actual requirement.
