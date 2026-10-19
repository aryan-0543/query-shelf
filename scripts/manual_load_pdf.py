"""Manually verify that a local PDF can be loaded."""

from pathlib import Path

from app.document_loader import DocumentLoadError, load_pdf

# Change this path to the PDF you want to test.
PDF_PATH = Path("Data/sample_pdfs/Clean Code ( PDFDrive.com ).pdf")


def main() -> None:
    """Load the configured PDF and report the result."""

    document = None

    try:
        document = load_pdf(PDF_PATH)
        print(f"PDF loaded successfully: {PDF_PATH}")
        print(f"Page count: {document.page_count}")
    except DocumentLoadError as error:
        print(f"PDF loading failed: {error}")
    finally:
        if document is not None:
            document.close()


if __name__ == "__main__":
    main()
