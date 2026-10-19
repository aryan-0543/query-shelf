from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from app.pdf_extractor import PageText, TextExtractionError, extract_text


def create_pdf(path: Path) -> None:
    document = pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "First page text")
    document.new_page()
    third_page = document.new_page()
    third_page.insert_text((72, 72), "Third page text")
    document.save(path)
    document.close()


def test_extract_text_returns_page_text_in_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    create_pdf(pdf_path)
    document = pymupdf.open(pdf_path)

    try:
        assert extract_text(document) == [
            PageText(page_number=1, text="First page text\n"),
            PageText(page_number=2, text=""),
            PageText(page_number=3, text="Third page text\n"),
        ]
    finally:
        document.close()


def test_extract_text_preserves_empty_pages() -> None:
    document = pymupdf.open()
    document.new_page()

    try:
        assert extract_text(document) == [PageText(page_number=1, text="")]
    finally:
        document.close()


def test_extract_text_reports_page_extraction_errors() -> None:
    failing_page = SimpleNamespace(
        get_text=lambda _format: (_ for _ in ()).throw(RuntimeError("failure"))
    )
    document = SimpleNamespace(page_count=1, load_page=lambda _index: failing_page)

    with pytest.raises(TextExtractionError, match="page 1"):
        extract_text(document)
