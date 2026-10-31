from pathlib import Path

import numpy as np
import pytest

from app.chunker import TextChunk
from app.embeddings import EXPECTED_EMBEDDING_DIMENSION, EmbeddedChunk
from app.vector_store import (
    VectorStore,
    VectorStoreError,
    chunk_id,
)


def embedded_chunk(
    *,
    document_id: str = "document",
    page_numbers: tuple[int, ...] = (2,),
    chunk_index: int = 0,
    token_count: int = 4,
    text: str = "chunk text",
    dimension: int = EXPECTED_EMBEDDING_DIMENSION,
    embedding: np.ndarray | None = None,
) -> EmbeddedChunk:
    chunk = TextChunk(
        document_id,
        page_numbers,
        chunk_index,
        text,
        token_count,
    )
    if embedding is None:
        embedding = np.arange(dimension, dtype=np.float32)
    return EmbeddedChunk(chunk, embedding)


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


def test_same_chunk_update_preserves_document_identity(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    item = embedded_chunk()

    store.store_chunks([item], content_fingerprint="fingerprint")
    store.store_chunks([item], content_fingerprint="fingerprint")

    result = store.get_by_id(chunk_id(item))
    assert result is not None
    assert result.content_fingerprint == "fingerprint"


def test_changed_document_replaces_stale_chunks(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    first = embedded_chunk(chunk_index=0)
    second = embedded_chunk(chunk_index=1, page_numbers=(3,))

    assert store.replace_document("document", "old", [first, second])
    assert store.replace_document("document", "new", [first]) is True
    assert store.get_by_id(chunk_id(second)) is None
    assert store.get_by_id(chunk_id(first)).content_fingerprint == "new"


def test_unchanged_document_skips_replacement(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    item = embedded_chunk()

    assert store.replace_document("document", "same", [item])
    assert store.replace_document("document", "same", [item]) is False
    assert store.is_current_document("document", "same")


def test_document_stats_are_scoped_to_document_and_preserve_records(
    tmp_path: Path,
) -> None:
    store = VectorStore(tmp_path / "chroma")
    first = embedded_chunk(document_id="first", chunk_index=0)
    second = embedded_chunk(document_id="first", chunk_index=1)
    other = embedded_chunk(document_id="other", chunk_index=0)
    store.replace_document("first", "fingerprint", [first, second])
    store.replace_document("other", "other-fingerprint", [other])

    stats = store.get_document_stats("first")

    assert stats.chunk_count == 2
    assert stats.embedding_dimension == EXPECTED_EMBEDDING_DIMENSION
    assert stats.content_fingerprint == "fingerprint"
    assert stats.chunk_ids == (chunk_id(first), chunk_id(second))
    assert store.get_by_id(chunk_id(first)) is not None
    assert store.get_by_id(chunk_id(second)) is not None
    assert store.get_by_id(chunk_id(other)) is not None


def test_invalid_embedding_dimension_is_rejected(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")

    with pytest.raises(VectorStoreError, match="Expected embedding dimension"):
        store.store_chunks([embedded_chunk(dimension=10)])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_embedding_is_rejected(tmp_path: Path, value: float) -> None:
    item = embedded_chunk()
    item.embedding[0] = value

    with pytest.raises(VectorStoreError, match="finite"):
        VectorStore(tmp_path / "chroma").store_chunks([item])


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

    with pytest.raises(ValueError, match=message):
        store.store_chunks([embedded_chunk(**values)])


def test_similarity_search_returns_ranked_metadata_and_distances(
    tmp_path: Path,
) -> None:
    store = VectorStore(tmp_path / "chroma")
    first_embedding = np.zeros(EXPECTED_EMBEDDING_DIMENSION, dtype=np.float32)
    first_embedding[0] = 1
    second_embedding = np.zeros(EXPECTED_EMBEDDING_DIMENSION, dtype=np.float32)
    second_embedding[1] = 1
    first = embedded_chunk(
        document_id="first",
        page_numbers=(1, 2),
        text="nearest",
        token_count=2,
        embedding=first_embedding,
    )
    second = embedded_chunk(
        document_id="second",
        page_numbers=(3,),
        text="next",
        token_count=3,
        embedding=second_embedding,
    )
    store.store_chunks([first, second])

    results = store.similarity_search(first_embedding, top_k=2)

    assert [result.chunk_id for result in results] == [
        chunk_id(first),
        chunk_id(second),
    ]
    assert results[0].text == "nearest"
    assert results[0].document_id == "first"
    assert results[0].page_numbers == [1, 2]
    assert results[0].chunk_index == 0
    assert results[0].token_count == 2
    assert results[0].distance == pytest.approx(0.0)
    assert results[1].distance > results[0].distance


def test_similarity_search_limits_results_and_handles_fewer_records(
    tmp_path: Path,
) -> None:
    store = VectorStore(tmp_path / "chroma")
    store.store_chunks([embedded_chunk()])

    results = store.similarity_search(np.ones(EXPECTED_EMBEDDING_DIMENSION), top_k=5)

    assert len(results) == 1


def test_similarity_search_empty_collection_returns_no_results(
    tmp_path: Path,
) -> None:
    store = VectorStore(tmp_path / "chroma")

    assert store.similarity_search(np.ones(EXPECTED_EMBEDDING_DIMENSION)) == []


@pytest.mark.parametrize(
    "embedding",
    [
        np.ones((2, EXPECTED_EMBEDDING_DIMENSION)),
        np.ones(EXPECTED_EMBEDDING_DIMENSION - 1),
        ["not", "numeric"],
        np.full(EXPECTED_EMBEDDING_DIMENSION, np.nan),
        np.full(EXPECTED_EMBEDDING_DIMENSION, np.inf),
    ],
)
def test_similarity_search_rejects_invalid_embeddings(
    tmp_path: Path, embedding: object
) -> None:
    store = VectorStore(tmp_path / "chroma")

    with pytest.raises(VectorStoreError):
        store.similarity_search(embedding)


@pytest.mark.parametrize("top_k", [True, False, 0, -1, 1.5, "5"])
def test_similarity_search_rejects_invalid_top_k(
    tmp_path: Path, top_k: object
) -> None:
    store = VectorStore(tmp_path / "chroma")

    with pytest.raises(VectorStoreError, match="top_k"):
        store.similarity_search(np.ones(EXPECTED_EMBEDDING_DIMENSION), top_k=top_k)
