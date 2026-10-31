from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from app.embeddings import EmbeddingError
from app.query import UserQuery
from app.retriever import (
    RetrievedChunk,
    Retriever,
    RetrievalResult,
    RetrieverError,
)
from app.vector_store import SimilaritySearchResult, VectorStoreError


def search_result(chunk_id: str = "chunk-1") -> SimilaritySearchResult:
    return SimilaritySearchResult(
        chunk_id=chunk_id,
        text="retrieved text",
        document_id="document",
        page_numbers=[2],
        chunk_index=3,
        token_count=12,
        distance=0.25,
    )


class FakeStore:
    def __init__(self, results=None, error=None) -> None:
        self.results = results if results is not None else [search_result()]
        self.error = error
        self.calls: list[tuple[object, int]] = []

    def similarity_search(self, embedding, *, top_k):
        self.calls.append((embedding, top_k))
        if self.error:
            raise self.error
        return self.results


def test_retrieve_orchestrates_embedding_search_and_ranks_results(monkeypatch) -> None:
    embedding = np.ones(384)
    store = FakeStore([search_result("first"), search_result("second")])
    calls = []

    def fake_embed(query):
        calls.append(query)
        return embedding

    monkeypatch.setattr("app.retriever.embed_query", fake_embed)

    result = Retriever(store).retrieve(UserQuery("What is clean code?"), top_k=2)

    assert calls == [result.query]
    assert store.calls == [(embedding, 2)]
    assert [chunk.rank for chunk in result.chunks] == [1, 2]
    assert result.total_chunks == 2
    assert result.chunks[0].page_numbers == (2,)
    assert result.chunks[0].distance == 0.25
    assert result.query.text == "What is clean code?"


def test_retrieve_uses_default_top_k(monkeypatch) -> None:
    store = FakeStore()
    monkeypatch.setattr("app.retriever.embed_query", lambda query: np.ones(384))

    Retriever(store).retrieve(UserQuery("question"))

    assert store.calls[0][1] == 5


def test_empty_results_are_valid(monkeypatch) -> None:
    store = FakeStore([])
    monkeypatch.setattr("app.retriever.embed_query", lambda query: np.ones(384))

    result = Retriever(store).retrieve(UserQuery("question"))

    assert result.chunks == ()
    assert result.total_chunks == 0


def test_duplicate_chunk_ids_are_rejected(monkeypatch) -> None:
    store = FakeStore([search_result(), search_result()])
    monkeypatch.setattr("app.retriever.embed_query", lambda query: np.ones(384))

    with pytest.raises(RetrieverError, match="duplicate"):
        Retriever(store).retrieve(UserQuery("question"))


@pytest.mark.parametrize(
    "result",
    [
        SimilaritySearchResult("", "text", "document", [1], 0, 1, 0.1),
        SimilaritySearchResult("id", " ", "document", [1], 0, 1, 0.1),
        SimilaritySearchResult("id", "text", "", [1], 0, 1, 0.1),
        SimilaritySearchResult("id", "text", "document", [], 0, 1, 0.1),
        SimilaritySearchResult("id", "text", "document", [1], -1, 1, 0.1),
        SimilaritySearchResult("id", "text", "document", [1], 0, -1, 0.1),
        SimilaritySearchResult("id", "text", "document", [1], 0, 1, float("nan")),
    ],
)
def test_malformed_results_are_rejected(monkeypatch, result) -> None:
    monkeypatch.setattr("app.retriever.embed_query", lambda query: np.ones(384))

    with pytest.raises(RetrieverError, match="malformed"):
        Retriever(FakeStore([result])).retrieve(UserQuery("question"))


@pytest.mark.parametrize("top_k", [True, False, 0, -1, 1.5, "5"])
def test_invalid_top_k_is_rejected(monkeypatch, top_k) -> None:
    monkeypatch.setattr("app.retriever.embed_query", lambda query: np.ones(384))

    with pytest.raises(RetrieverError, match="top_k"):
        Retriever(FakeStore()).retrieve(UserQuery("question"), top_k=top_k)


def test_invalid_query_is_rejected() -> None:
    with pytest.raises(RetrieverError, match="UserQuery"):
        Retriever(FakeStore()).retrieve("question")  # type: ignore[arg-type]


def test_embedding_failure_is_chained(monkeypatch) -> None:
    error = EmbeddingError("failed")
    monkeypatch.setattr(
        "app.retriever.embed_query",
        lambda query: (_ for _ in ()).throw(error),
    )

    with pytest.raises(RetrieverError) as raised:
        Retriever(FakeStore()).retrieve(UserQuery("question"))

    assert raised.value.__cause__ is error


def test_vector_store_failure_is_chained(monkeypatch) -> None:
    error = VectorStoreError("failed")
    monkeypatch.setattr("app.retriever.embed_query", lambda query: np.ones(384))

    with pytest.raises(RetrieverError) as raised:
        Retriever(FakeStore(error=error)).retrieve(UserQuery("question"))

    assert raised.value.__cause__ is error


def test_timings_are_measured_separately(monkeypatch) -> None:
    clock = iter([10.0, 10.25, 20.0, 20.5])
    monkeypatch.setattr("app.retriever.perf_counter", lambda: next(clock))
    monkeypatch.setattr("app.retriever.embed_query", lambda query: np.ones(384))

    result = Retriever(FakeStore()).retrieve(UserQuery("question"))

    assert result.query_embedding_seconds == pytest.approx(0.25)
    assert result.vector_search_seconds == pytest.approx(0.5)


def test_result_dataclasses_are_immutable() -> None:
    chunk = RetrievedChunk("id", "text", "doc", (1,), 0, 1, 0.1, 1)
    result = RetrievalResult(UserQuery("question"), (chunk,), 0.1, 0.2)

    with pytest.raises(FrozenInstanceError):
        chunk.rank = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.query = UserQuery("other")  # type: ignore[misc]
    assert isinstance(result.chunks, tuple)
