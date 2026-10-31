"""Orchestrate query embedding and candidate chunk retrieval."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from app.embeddings import EmbeddingError, embed_query
from app.query import UserQuery
from app.vector_store import (
    SimilaritySearchResult,
    VectorStore,
    VectorStoreError,
)


class RetrieverError(Exception):
    """Raised when candidate retrieval cannot produce a valid result."""


@dataclass(frozen=True)
class RetrievedChunk:
    """An immutable, traceable candidate chunk."""

    chunk_id: str
    text: str
    document_id: str
    page_numbers: tuple[int, ...]
    chunk_index: int
    token_count: int
    distance: float
    rank: int


@dataclass(frozen=True)
class RetrievalResult:
    """The immutable result of candidate retrieval."""

    query: UserQuery
    chunks: tuple[RetrievedChunk, ...]
    query_embedding_seconds: float
    vector_search_seconds: float

    @property
    def total_chunks(self) -> int:
        """Return the number of retrieved candidate chunks."""

        return len(self.chunks)


class Retriever:
    """Coordinate query embedding and native vector search."""

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: UserQuery,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Embed a validated query and return ranked candidate chunks."""

        if not isinstance(query, UserQuery):
            raise RetrieverError("Query must be a validated UserQuery")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrieverError("top_k must be a positive integer")

        embedding_started = perf_counter()
        try:
            query_embedding = embed_query(query)
        except EmbeddingError as error:
            raise RetrieverError("Unable to embed query") from error
        embedding_elapsed = perf_counter() - embedding_started

        search_started = perf_counter()
        try:
            results = self._vector_store.similarity_search(
                query_embedding,
                top_k=top_k,
            )
        except VectorStoreError as error:
            raise RetrieverError("Unable to search vector store") from error
        search_elapsed = perf_counter() - search_started

        try:
            chunks = tuple(
                self._to_retrieved_chunk(result, rank)
                for rank, result in enumerate(results, start=1)
            )
        except (TypeError, ValueError, KeyError) as error:
            raise RetrieverError("Vector search returned malformed results") from error

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RetrieverError("Vector search returned duplicate chunk IDs")

        return RetrievalResult(
            query=query,
            chunks=chunks,
            query_embedding_seconds=embedding_elapsed,
            vector_search_seconds=search_elapsed,
        )

    @staticmethod
    def _to_retrieved_chunk(
        result: SimilaritySearchResult,
        rank: int,
    ) -> RetrievedChunk:
        if not isinstance(result, SimilaritySearchResult):
            raise TypeError("Search result has an unexpected type")
        if not isinstance(result.chunk_id, str) or not result.chunk_id.strip():
            raise ValueError("Search result chunk_id must not be empty")
        if not isinstance(result.text, str) or not result.text.strip():
            raise ValueError("Search result text must not be empty")
        if not isinstance(result.document_id, str) or not result.document_id.strip():
            raise ValueError("Search result document_id must not be empty")
        if not result.page_numbers or any(
            isinstance(page, bool) or not isinstance(page, int) or page <= 0
            for page in result.page_numbers
        ):
            raise ValueError("Search result page_numbers are invalid")
        if (
            isinstance(result.chunk_index, bool)
            or not isinstance(result.chunk_index, int)
            or result.chunk_index < 0
        ):
            raise ValueError("Search result chunk_index is invalid")
        if (
            isinstance(result.token_count, bool)
            or not isinstance(result.token_count, int)
            or result.token_count < 0
        ):
            raise ValueError("Search result token_count is invalid")
        if not isinstance(result.distance, (int, float, np.number)) or not np.isfinite(
            result.distance
        ):
            raise ValueError("Search result distance is invalid")
        return RetrievedChunk(
            chunk_id=result.chunk_id,
            text=result.text,
            document_id=result.document_id,
            page_numbers=tuple(result.page_numbers),
            chunk_index=result.chunk_index,
            token_count=result.token_count,
            distance=float(result.distance),
            rank=rank,
        )
