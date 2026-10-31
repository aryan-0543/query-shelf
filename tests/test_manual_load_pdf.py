from pathlib import Path
from types import SimpleNamespace

import numpy as np

import scripts.manual_load_pdf as manual_load_pdf
from app.embeddings import EmbeddedChunk, EXPECTED_EMBEDDING_DIMENSION
from app.pdf_extractor import PageText
from app.vector_store import DocumentStats


class FakeDocument:
    page_count = 1

    def close(self) -> None:
        pass


def run_manual_script(monkeypatch, *, unchanged: bool) -> str:
    embedded = EmbeddedChunk(
        chunk=SimpleNamespace(
            document_id="document",
            page_numbers=(1,),
            chunk_index=0,
            text="chunk",
            token_count=1,
        ),
        embedding=np.ones(EXPECTED_EMBEDDING_DIMENSION),
    )
    stats = DocumentStats(
        document_id="document",
        chunk_count=1,
        embedding_dimension=EXPECTED_EMBEDDING_DIMENSION,
        content_fingerprint="fingerprint",
        chunk_ids=("document::chunk::0",),
    )

    class FakeStore:
        def is_current_document(self, document_id, fingerprint):
            return unchanged

        def replace_document(self, document_id, fingerprint, chunks):
            return not unchanged

        def get_document_stats(self, document_id):
            return stats

        def get_by_id(self, chunk_id):
            return embedded if chunk_id in stats.chunk_ids else None

    monkeypatch.setattr(manual_load_pdf, "PDF_PATH", Path("sample.pdf"))
    monkeypatch.setattr(manual_load_pdf, "load_pdf", lambda path: FakeDocument())
    monkeypatch.setattr(
        manual_load_pdf,
        "extract_text",
        lambda document: [PageText(page_number=1, text="text")],
    )
    monkeypatch.setattr(manual_load_pdf, "document_id", lambda path: "document")
    monkeypatch.setattr(
        manual_load_pdf,
        "content_fingerprint",
        lambda path: "fingerprint",
    )
    monkeypatch.setattr(manual_load_pdf, "chunk_pages", lambda pages, document: [embedded.chunk])
    monkeypatch.setattr(
        manual_load_pdf,
        "embed_chunks",
        lambda chunks, batch_size: [embedded],
    )
    monkeypatch.setattr(manual_load_pdf, "VectorStore", FakeStore)

    manual_load_pdf.main()
    return ""


def test_manual_summary_reports_preserved_chunks_when_skipped(
    monkeypatch, capsys
) -> None:
    run_manual_script(monkeypatch, unchanged=True)

    output = capsys.readouterr().out
    assert "Document unchanged; re-indexing skipped." in output
    assert "Existing chunks are preserved in ChromaDB." in output
    assert "Chunks processed this run: 0" in output
    assert "Embeddings generated this run: 0" in output
    assert "Chunks already stored in ChromaDB: 1" in output
    assert "Chunks retrieved by ID: 1" in output


def test_manual_summary_reports_fresh_processing(monkeypatch, capsys) -> None:
    run_manual_script(monkeypatch, unchanged=False)

    output = capsys.readouterr().out
    assert "Unchanged re-index skipped: False" in output
    assert "Chunks processed this run: 1" in output
    assert "Embeddings generated this run: 1" in output
    assert "Chunks already stored in ChromaDB: 1" in output
    assert "Chunks retrieved by ID: 1" in output
