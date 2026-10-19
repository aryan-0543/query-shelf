from pathlib import Path

import pymupdf
import pytest

from app.document_loader import DocumentLoadError, load_pdf


def create_pdf(path: Path, text: str = "Test document") -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_load_pdf_returns_open_document(tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    create_pdf(pdf_path)

    document = load_pdf(pdf_path)

    try:
        assert document.is_pdf
        assert document.page_count == 1
    finally:
        document.close()


def test_load_pdf_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="does not exist"):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_rejects_non_pdf_file(tmp_path: Path) -> None:
    document_path = tmp_path / "document.txt"
    document_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Expected a \\.pdf file"):
        load_pdf(document_path)


def test_load_pdf_rejects_corrupted_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_bytes(b"not a valid PDF")

    with pytest.raises(DocumentLoadError, match="Unable to open PDF"):
        load_pdf(pdf_path)


def test_load_pdf_rejects_password_protected_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "protected.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(
        pdf_path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    document.close()

    with pytest.raises(DocumentLoadError, match="password-protected"):
        load_pdf(pdf_path)
