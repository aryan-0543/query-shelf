from pathlib import Path

from app.document_identity import content_fingerprint, document_id


def test_same_filename_in_different_paths_has_distinct_document_id(
    tmp_path: Path,
) -> None:
    first = tmp_path / "one" / "document.pdf"
    second = tmp_path / "two" / "document.pdf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same")
    second.write_bytes(b"same")

    assert document_id(first) != document_id(second)


def test_document_id_is_stable_and_fingerprint_changes_with_content(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"first")

    first_id = document_id(pdf_path)
    first_fingerprint = content_fingerprint(pdf_path)
    pdf_path.write_bytes(b"second")

    assert document_id(pdf_path) == first_id
    assert content_fingerprint(pdf_path) != first_fingerprint
