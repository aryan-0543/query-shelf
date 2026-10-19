
"""Manually verify PDF loading and page-level text extraction."""

from pathlib import Path

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

        print("Text extraction completed successfully.")
        print(f"Pages processed: {len(pages)}/{total_pages}")
        print(f"Pages containing text: {pages_with_text}")
        print(f"Empty pages: {empty_pages}")
        print(f"Total characters extracted: {total_characters}")

    except DocumentLoadError as error:
        print(f"PDF loading failed: {error}")
    except TextExtractionError as error:
        print(f"PDF text extraction failed: {error}")
    finally:
        if document is not None:
            document.close()
            print("PDF document closed.")


if __name__ == "__main__":
    main()
