
"""Manually verify PDF loading and page-level text extraction."""

from pathlib import Path
from time import perf_counter

from app.chunker import TextChunk, chunk_pages
from app.document_identity import content_fingerprint, document_id
from app.document_loader import DocumentLoadError, load_pdf
from app.embeddings import (
    DEFAULT_BATCH_SIZE,
    EmbeddingError,
    embed_chunks,
)
from app.pdf_extractor import TextExtractionError, extract_text
from app.vector_store import VectorStore, VectorStoreError, chunk_id

PDF_PATH = Path("Data/sample_pdfs/Clean Code ( PDFDrive.com ).pdf")


def main() -> None:
    """Load a PDF, extract text, and report summary statistics."""

    document = None

    try:
        print(f"Loading PDF: {PDF_PATH}")
        document = load_pdf(PDF_PATH)
        current_document_id = document_id(PDF_PATH)
        current_fingerprint = content_fingerprint(PDF_PATH)

        total_pages = document.page_count
        print("PDF loaded successfully.")
        print(f"Total pages: {total_pages}")
        print("\nStarting text extraction...")

        extraction_started = perf_counter()
        pages = extract_text(document)
        extraction_elapsed = perf_counter() - extraction_started

        pages_with_text = sum(bool(page.text.strip()) for page in pages)
        empty_pages = len(pages) - pages_with_text
        total_characters = sum(len(page.text) for page in pages)
        vector_store = VectorStore()
        unchanged_before_processing = vector_store.is_current_document(
            current_document_id,
            current_fingerprint,
        )
        chunking_started = perf_counter()
        chunks = (
            []
            if unchanged_before_processing
            else chunk_pages(pages, current_document_id)
        )
        chunking_elapsed = perf_counter() - chunking_started
        embedding_started = perf_counter()
        embedded_chunks = (
            []
            if unchanged_before_processing
            else embed_chunks(chunks, batch_size=DEFAULT_BATCH_SIZE)
        )
        embedding_elapsed = perf_counter() - embedding_started
        changed = False
        if not unchanged_before_processing:
            changed = vector_store.replace_document(
                current_document_id,
                current_fingerprint,
                embedded_chunks,
            )
        unchanged = not vector_store.replace_document(
            current_document_id,
            current_fingerprint,
            embedded_chunks,
        )
        retrieved_chunks = [
            vector_store.get_by_id(chunk_id(embedded_chunk))
            for embedded_chunk in embedded_chunks
        ]
        retrieved_count = (
            len(embedded_chunks)
            if unchanged_before_processing
            else sum(chunk is not None for chunk in retrieved_chunks)
        )

        print(
            f"Text extraction completed successfully "
            f"({extraction_elapsed:.2f}s)."
        )
        print(f"Pages processed: {len(pages)}/{total_pages}")
        print(f"Pages containing text: {pages_with_text}")
        print(f"Empty pages: {empty_pages}")
        print(f"Total characters extracted: {total_characters}")
        print(f"Chunking completed successfully ({chunking_elapsed:.2f}s).")
        print(f"Total chunks: {len(chunks)}")
        print(
            f"Embedding generation completed successfully "
            f"({embedding_elapsed:.2f}s)."
        )
        print(f"Total embeddings: {len(embedded_chunks)}")
        dimensions = (
            embedded_chunks[0].embedding.shape[0] if embedded_chunks else 0
        )
        print(f"Embedding dimensions: {dimensions}")
        print(f"Embedding batch size: {DEFAULT_BATCH_SIZE}")
        print(f"Document fingerprint: {current_fingerprint}")
        print(f"Document replacement performed: {changed}")
        print(
            f"Unchanged re-index skipped: "
            f"{unchanged_before_processing or unchanged}"
        )
        print(f"Chunks stored: {len(embedded_chunks)}")
        print(f"Chunks retrieved by ID: {retrieved_count}")
        print("\nProcessing summary:")
        print(
            f"  Pages: {total_pages} total, {pages_with_text} non-empty | "
            f"Chunks: {len(chunks)} | Embeddings: {len(embedded_chunks)} | "
            f"Dimensions: {dimensions} | Stored: {len(embedded_chunks)} | "
            f"Retrieved: {retrieved_count}"
        )
        print("Chunk metadata examples:")
        for embedded_chunk in embedded_chunks[:3]:
            print(f"  {chunk_metadata_summary(embedded_chunk.chunk)}")

    except DocumentLoadError as error:
        print(f"PDF loading failed: {error}")
    except TextExtractionError as error:
        print(f"PDF text extraction failed: {error}")
    except EmbeddingError as error:
        print(f"Embedding generation failed: {error}")
    except VectorStoreError as error:
        print(f"Vector storage failed: {error}")
    finally:
        if document is not None:
            document.close()
            print("PDF document closed.")


def chunk_metadata_summary(chunk: TextChunk) -> str:
    """Return metadata for a chunk without exposing its text."""

    return (
        f"document_id={chunk.document_id!r}, "
        f"pages={chunk.page_numbers}, "
        f"chunk_index={chunk.chunk_index}, "
        f"token_count={chunk.token_count}"
    )


if __name__ == "__main__":
    main()
