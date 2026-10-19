"""Load and validate local PDF documents."""
from pathlib import Path

import pymupdf


class DocumentLoadError(Exception):
    """Raised when a local PDF cannot be loaded safely."""


def load_pdf(path: str | Path) -> pymupdf.Document:
    """Open and return a readable, unencrypted PDF document.

    The caller owns the returned document and must close it when finished.
    Text extraction is intentionally outside this module.
    """

    pdf_path = Path(path)

    if not pdf_path.exists():
        raise DocumentLoadError(f"PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise DocumentLoadError(f"PDF path is not a file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise DocumentLoadError(f"Expected a .pdf file: {pdf_path}")

    try:
        document = pymupdf.open(str(pdf_path))
    except (pymupdf.FileDataError, OSError, RuntimeError) as error:
        raise DocumentLoadError(f"Unable to open PDF file: {pdf_path}") from error

    if document.needs_pass:
        document.close()
        raise DocumentLoadError(
            f"PDF is password-protected and cannot be opened without a password: "
            f"{pdf_path}"
        )

    return document
