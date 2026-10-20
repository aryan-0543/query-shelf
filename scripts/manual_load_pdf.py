
"""Manually verify PDF loading and page-level text extraction."""

from pathlib import Path

from app.chunker import TextChunk, chunk_pages
from app.document_loader import DocumentLoadError, load_pdf
from app.pdf_extractor import TextExtractionError, extract_text

PDF_PATH = Path("Data/sample_pdfs/Clean Code ( PDFDrive.com ).pdf")


def main() -> None:
    """Load a PDF, extract text, and report summary statistics."""

    document = None

    try:
        print(f"Loading PDF: {PDF_PATH}")
        document = load_pdf(PDF_PATH)

        total_pages = document.page_count
        print("PDF loaded successfully.")
        print(f"Total pages: {total_pages}")
        print("\nStarting text extraction...")

        pages = extract_text(document)

        pages_with_text = sum(bool(page.text.strip()) for page in pages)
        empty_pages = len(pages) - pages_with_text
        total_characters = sum(len(page.text) for page in pages)
        chunks = chunk_pages(pages, PDF_PATH.stem)

        print("Text extraction completed successfully.")
        print(f"Pages processed: {len(pages)}/{total_pages}")
        print(f"Pages containing text: {pages_with_text}")
        print(f"Empty pages: {empty_pages}")
        print(f"Total characters extracted: {total_characters}")
        print(f"Total chunks: {len(chunks)}")
        print("Chunk metadata examples:")
        for chunk in chunks[:3]:
            print(
                f"  {chunk_metadata_summary(chunk)}"
            )

    except DocumentLoadError as error:
        print(f"PDF loading failed: {error}")
    except TextExtractionError as error:
        print(f"PDF text extraction failed: {error}")
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
