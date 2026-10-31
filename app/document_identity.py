"""Stable PDF identity and content fingerprint helpers."""

import hashlib
from pathlib import Path


def document_id(path: str | Path) -> str:
    """Return a stable ID derived from the PDF's resolved path."""

    resolved_path = str(Path(path).resolve())
    return hashlib.sha256(resolved_path.encode("utf-8")).hexdigest()


def content_fingerprint(path: str | Path) -> str:
    """Return the SHA-256 fingerprint of a PDF's contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
