"""Extract page-level text from an already loaded PDF document."""

from dataclasses import dataclass

import pymupdf


class TextExtractionError(Exception):
    """Raised when text cannot be extracted from a PDF page."""


@dataclass(frozen=True)
class PageText:
    """Text extracted from one PDF page."""

    page_number: int
    text: str


def extract_text(document: pymupdf.Document) -> list[PageText]:
    """Extract text from every page in an open PDF document.

    Page numbers are one-based for user-facing citations. Empty pages are
    retained with an empty text value. The caller owns and closes ``document``.
    """

    pages: list[PageText] = []

    for page_index in range(document.page_count):
        page_number = page_index + 1

        try:
            page = document.load_page(page_index)
            text = page.get_text("text")
        except (RuntimeError, ValueError) as error:
            raise TextExtractionError(
                f"Unable to extract text from PDF page {page_number}"
            ) from error

        pages.append(PageText(page_number=page_number, text=text))

    return pages
