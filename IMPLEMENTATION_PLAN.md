# Implementation Plan

## Status conventions

Every milestone below is **Pending** until its implementation and stated
verification criteria have actually been completed. The project structure may
evolve; any feature-driven additions should be proposed and explained before
implementation, with related documentation updated.

## 1. Project setup and environment

**Status:** Pending

### Objective

Establish the Python project, dependency installation, test foundation, and
protected local configuration.

### Expected files

```text
requirements.txt
.env.example
.gitignore
app/__init__.py
tests/
```

The exact structure may change if implementation needs justify a better
boundary.

### Implementation tasks

- Confirm a Python version compatible with the selected dependencies.
- Maintain `requirements.txt` with only required initial dependencies:
  PyMuPDF, `sentence-transformers`, ChromaDB, `google-genai`,
  `python-dotenv`, and pytest.
- Load `GEMINI_API_KEY` and `GEMINI_MODEL` from environment variables.
- Provide a clear missing/invalid-key error without printing the key.
- Use `.env.example` for placeholders only.
- Keep `.env` ignored by Git and never create it with invented credentials.
- Ignore virtual environments, caches, generated ChromaDB data, and other
  local artifacts that should not be committed.

### Tests

- Configuration loads expected values.
- Missing `GEMINI_API_KEY` gives a helpful error.
- Secrets do not appear in errors or logs.
- The example environment file contains no real credential.

### Completion criteria

- Dependencies install in a fresh virtual environment.
- A local `.env` can be created from `.env.example`.
- Configuration and secret-protection tests pass.
- No real secret is added to source, documentation, fixtures, or Git history.

## 2. Document loading

**Status:** Complete

### Objective

Validate and open one local PDF while keeping file loading separate from text
extraction and all later RAG processing.

### Expected files

```text
app/document_loader.py
scripts/manual_load_pdf.py
tests/test_document_loader.py
```

### Implementation tasks

- Accept a `str` or `Path` PDF path.
- Verify that the path exists and is a regular file.
- Verify that the filename has a `.pdf` extension.
- Open the file with PyMuPDF.
- Reject corrupted or unreadable PDFs with a clear error.
- Reject password-protected PDFs without attempting to bypass protection.
- Return an open PyMuPDF document and make resource ownership explicit.
- Keep extraction, chunking, embeddings, vector storage, and CLI behavior out
  of this milestone.
- Provide a manually configured verification script that reports successful
  loading and page count or a clear loading error.
- Close the document in the manual verification script.
- Run the manual script from the repository root as
  `python -m scripts.manual_load_pdf`.

### Tests

- A valid PDF returns an open document with the expected page count.
- Missing paths produce a clear error.
- Non-PDF files are rejected.
- Corrupted PDFs are rejected.
- Password-protected PDFs are rejected.
- The manual verification script reports the result and closes a successfully
  loaded document.

### Completion criteria

The focused document-loader tests pass, and the module contains no text
extraction or downstream RAG logic.

## 3. PDF text extraction

**Status:** Complete

### Objective

Extract ordered, page-aware text from one local PDF.

### Expected files

```text
app/pdf_extractor.py
scripts/manual_load_pdf.py
tests/test_pdf_extractor.py
```

### Implementation tasks

- Accept the open PyMuPDF document returned by `load_pdf()`.
- Extract text page by page without opening or closing the document.
- Return ordered `PageText` records with one-based page numbers.
- Preserve empty pages as records with empty text.
- Raise a page-specific `TextExtractionError` when extraction fails.
- Extend the central manual verification script to display extraction
  progress and each page's text while safely closing the document.

### Tests

- Successful extraction from a multi-page PDF.
- Page ordering and one-based page numbers.
- Empty-page behavior.
- Page extraction error handling.
- Manual extraction output and error handling.

### Completion criteria

Given an open PDF, extraction returns deterministic page-level records with
traceable page numbers, preserves empty pages, and reports extraction errors
explicitly. Focused extraction tests and the existing loader tests pass.

## 4. Chunking

**Status:** Complete

### Objective

Convert page text into deterministic retrieval chunks without losing source
information.

### Expected files

```text
app/chunker.py
tests/test_chunker.py
```

### Implementation tasks

- Implement custom Python chunking.
- Make chunk size and overlap configurable.
- Preserve page/source metadata.
- Define empty and short-text behavior.

### Tests

- Chunk size constraints.
- Overlap behavior.
- Ordering.
- Metadata preservation.
- Empty input.

### Completion criteria

Chunking is deterministic, independently testable, and every chunk can be
traced to its source/page. Chunk construction validates required metadata while
allowing short and heading-only chunks.

## 5. Embedding generation

**Status:** Complete

### Objective

Generate local vectors for document chunks and user queries using
`all-MiniLM-L6-v2`.

### Expected files

```text
app/embeddings.py
tests/test_embeddings.py
```

### Implementation tasks

- Load the sentence-transformers model locally.
- Implement document and query embedding paths.
- Ensure both paths use the same model and compatible dimensions.
- Provide a boundary that can be mocked in tests.

### Tests

- Mocked embedding behavior.
- Consistent vector dimensions.
- Empty-input behavior.
- Document/query compatibility.

### Completion criteria

Local document and query embeddings work through one understandable,
testable interface without Gemini API calls. Shared model configuration and
finite-value validation are implemented.

## 6. Vector database storage

**Status:** Complete

### Objective

Persist chunks, vectors, and metadata in local ChromaDB.

### Expected files

```text
app/vector_store.py
tests/test_vector_store.py
```

### Implementation tasks

- Create/open a local persistent collection at the approved configurable
  location, defaulting to `data/chroma/`.
- Generate deterministic chunk IDs.
- Store text, vectors, and source/page metadata.
- Define collection replacement or reset behavior.
- Store existing `EmbeddedChunk` vectors without regenerating embeddings.
- Validate 384-dimensional vectors and required chunk metadata.
- Reject duplicate IDs in a batch and conflicting existing records, while
  allowing safe updates for the same document/chunk.
- Serialize page-number metadata for ChromaDB and restore it during retrieval.
- Derive stable path-based document IDs separately from SHA-256 content
  fingerprints.
- Skip unchanged re-indexing and replace changed-document chunks, removing
  stale records.
- Check the stored fingerprint before chunking and embedding so unchanged
  documents can skip those stages.

### Tests

- Add and retrieve records.
- Metadata persistence.
- Collection isolation.
- Query result shape.
- Temporary test storage.

### Completion criteria

Indexed chunks persist, can be retrieved by ID, and can be updated safely
without invoking the embedding model or Gemini. Changed documents replace
their previous chunk set, while different document IDs remain isolated.

## 7. Query retrieval

**Status:** Pending

### Objective

Retrieve the most relevant chunks for a user query.

### Expected files

```text
app/retriever.py
tests/test_retriever.py
```

### Implementation tasks

- Embed queries locally.
- Search ChromaDB by vector similarity.
- Support configurable `top_k`.
- Return ranked chunks and source metadata.
- Define empty-collection and no-result behavior.

### Tests

- Mock query embeddings.
- Controlled-vector ranking.
- `top_k` behavior.
- Metadata propagation.
- Empty-result handling.

### Completion criteria

A query returns ranked, traceable chunks without calling Gemini.

## 8. Prompt augmentation and construction

**Status:** Pending

### Objective

Create a grounded prompt from the question and retrieved context.

### Expected files

```text
app/prompt_builder.py
tests/test_prompt_builder.py
```

### Implementation tasks

- Format retrieved chunks with source/page labels.
- Include the user question.
- Instruct the model to use only supplied context.
- Instruct it to acknowledge insufficient context.
- Define empty-context behavior.

### Tests

- Context inclusion.
- Source inclusion.
- Question inclusion.
- Empty-context behavior.
- Confirm that the complete PDF is not implicitly included.

### Completion criteria

Prompt construction is deterministic and independently testable.

## 9. Gemini API integration

**Status:** Pending

### Objective

Integrate the official Google Gen AI Python SDK behind an isolated provider
boundary.

### Expected files

```text
app/gemini_client.py
tests/test_gemini_client.py
```

### Implementation tasks

- Create the client from validated environment configuration.
- Use the configurable `GEMINI_MODEL`.
- Verify model availability and generation compatibility with the user's API
  key before finalizing a default.
- Record applicable free-tier quotas and rate limits after verification.
- Mock Gemini responses in ordinary unit tests.
- Handle missing credentials, invalid credentials, quota errors, and network
  errors clearly.
- Never log or include API keys in errors.
- Send only the question and retrieved context required for generation.

### Tests

- Mocked successful response.
- Mocked provider error.
- Missing-key behavior.
- Invalid-configuration behavior.
- Request-content verification.
- Secret-redaction verification.

### Completion criteria

Gemini usage is isolated to one module, ordinary tests make no network calls,
and a separate credential-dependent check verifies the selected model.

## 10. End-to-end pipeline

**Status:** Pending

### Objective

Connect both RAG phases—document indexing and retrieval/generation—in the CLI.

### Expected files

```text
app/pipeline.py
app/cli.py
tests/test_pipeline.py
```

### Implementation tasks

- Orchestrate document loading, text extraction, chunking, local embedding
  generation, and vector storage for indexing.
- Orchestrate query embedding, vector search, retrieval, prompt augmentation,
  Gemini generation, and response formatting.
- Implement the interactive question loop first.
- Display answer and available page/source references.
- Surface missing-file, empty-extraction, configuration, and provider errors.
- Defer one-shot question mode to a later milestone.

### Tests

- Mock component boundaries.
- Verify call order.
- Verify source references reach CLI output.
- Verify failure propagation.
- Test interactive-loop exit behavior.

### Completion criteria

A mocked end-to-end test passes without network access, and the interactive
CLI is manually verified with a user-provided PDF and API key.

## 11. Testing and retrieval evaluation

**Status:** Pending

### Objective

Measure extraction, retrieval, grounding, and error behavior with reproducible
tests and a small reviewed evaluation set.

### Expected files

Potentially:

```text
tests/
evaluation/
```

These directories and files should be added only when the evaluation work
requires them.

### Implementation tasks

- Add regression tests for extraction and chunking.
- Add retrieval fixtures with expected chunks/pages.
- Create a small manually reviewed question set.
- Measure expected-source retrieval at selected top-k values.
- Test unsupported questions and weak/empty context.
- Keep live Gemini smoke checks separate from default unit tests.
- Record quota-conscious testing practices.

### Tests and evaluation

- Automated unit and integration tests.
- Small retrieval evaluation set.
- Optional explicit live Gemini smoke test.

### Completion criteria

The test suite is reproducible, retrieval quality has concrete examples, and
live API checks are isolated and quota-aware.

## Deferred one-shot CLI milestone

One-shot question mode is intentionally planned after the interactive loop.
Before implementing it, the CLI shape and documentation should be revisited
to decide whether one command should support both an interactive default and a
single-question argument. Any new parser or module boundary should be
proposed first if it affects the documented structure.

## Change-control rule for future structure

The listed files are not a promise that the project will never grow beyond
them. When a new feature needs a module, folder, or configuration file:

1. Explain the concrete requirement and why existing boundaries are
   insufficient.
2. Propose the new structure before implementation.
3. Update this plan and any affected architecture/README sections.
4. Add only the smallest structure that solves the requirement.
