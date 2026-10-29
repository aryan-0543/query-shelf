from pathlib import Path

import numpy as np
import pytest

from app.chunker import TextChunk
from app.embeddings import EXPECTED_EMBEDDING_DIMENSION, EmbeddedChunk
from app.vector_store import VectorStore, VectorStoreError, chunk_id


def embedded_chunk(
    *,
    document_id: str = "document",
    page_numbers: tuple[int, ...] = (2,),
    chunk_index: int = 0,
    token_count: int = 4,
    text: str = "chunk text",
    dimension: int = EXPECTED_EMBEDDING_DIMENSION,
) -> EmbeddedChunk:
    chunk = TextChunk(
        document_id,
        page_numbers,
        chunk_index,
        text,
        token_count,
    )
    return EmbeddedChunk(chunk, np.arange(dimension, dtype=np.float32))


def test_store_and_retrieve_chunk(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    item = embedded_chunk()

    ids = store.store_chunks([item])
    result = store.get_by_id(ids[0])

    assert ids == [chunk_id(item)]
    assert result is not None
    assert result.text == item.chunk.text
    assert result.document_id == item.chunk.document_id
    assert result.page_numbers == [2]
    assert result.chunk_index == 0
    assert result.token_count == 4
    np.testing.assert_array_equal(result.embedding, item.embedding)


def test_chunks_persist_across_store_instances(tmp_path: Path) -> None:
    storage_path = tmp_path / "chroma"
    item = embedded_chunk()
    first_store = VectorStore(storage_path)
    first_store.store_chunks([item])

    result = VectorStore(storage_path).get_by_id(chunk_id(item))

    assert result is not None
    assert result.page_numbers == [2]


def test_same_chunk_can_be_updated(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    original = embedded_chunk()
    updated = embedded_chunk(text="updated text", token_count=5)

    store.store_chunks([original])
    store.store_chunks([updated])

    result = store.get_by_id(chunk_id(original))
    assert result is not None
    assert result.text == "updated text"
    assert result.token_count == 5


def test_duplicate_ids_in_batch_are_rejected(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    item = embedded_chunk()

    with pytest.raises(VectorStoreError, match="Duplicate chunk IDs"):
        store.store_chunks([item, item])


def test_conflicting_existing_record_is_rejected(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    item = embedded_chunk()
    store._collection.upsert(
        ids=[chunk_id(item)],
        documents=["conflicting"],
        embeddings=[item.embedding.tolist()],
        metadatas=[
            {
                "document_id": "other-document",
                "page_numbers": "2",
                "chunk_index": 0,
                "token_count": 4,
            }
        ],
    )

    with pytest.raises(VectorStoreError, match="conflicts"):
        store.store_chunks([item])


def test_invalid_embedding_dimension_is_rejected(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")

    with pytest.raises(VectorStoreError, match="Expected embedding dimension"):
        store.store_chunks([embedded_chunk(dimension=10)])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("document_id", "", "document_id"),
        ("page_numbers", (), "page_numbers"),
        ("chunk_index", -1, "chunk_index"),
        ("token_count", -1, "token_count"),
        ("text", " ", "text"),
    ],
)
def test_required_metadata_is_validated(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    values = {
        "document_id": "document",
        "page_numbers": (2,),
        "chunk_index": 0,
        "token_count": 4,
        "text": "chunk text",
    }
    values[field] = value
    store = VectorStore(tmp_path / "chroma")

    with pytest.raises(VectorStoreError, match=message):
        store.store_chunks([embedded_chunk(**values)])
