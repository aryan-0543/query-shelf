"""Persist embedded document chunks in a local ChromaDB collection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from app.embeddings import EXPECTED_EMBEDDING_DIMENSION, EmbeddedChunk

DEFAULT_STORAGE_PATH = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "document_chunks"


class VectorStoreError(Exception):
    """Raised when vector storage cannot validate or persist a chunk."""


@dataclass(frozen=True)
class StoredChunk:
    """A chunk record retrieved from ChromaDB."""

    chunk_id: str
    text: str
    embedding: np.ndarray
    document_id: str
    page_numbers: list[int]
    chunk_index: int
    token_count: int


def chunk_id(embedded_chunk: EmbeddedChunk) -> str:
    """Return a deterministic ID for a document chunk."""

    return (
        f"{embedded_chunk.chunk.document_id}::chunk::"
        f"{embedded_chunk.chunk.chunk_index}"
    )


class VectorStore:
    """Store and retrieve embedded chunks in persistent local storage."""

    def __init__(
        self,
        storage_path: str | Path = DEFAULT_STORAGE_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        self._client = chromadb.PersistentClient(path=str(storage_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )

    def store_chunks(self, chunks: list[EmbeddedChunk]) -> list[str]:
        """Store new chunks and safely update existing matching chunks."""

        records = [self._validated_record(chunk) for chunk in chunks]
        ids = [record[0] for record in records]
        if len(ids) != len(set(ids)):
            raise VectorStoreError("Duplicate chunk IDs in storage batch")
        if not records:
            return []

        existing = self._collection.get(ids=ids, include=["metadatas"])
        existing_metadata = dict(
            zip(existing["ids"], existing.get("metadatas", []), strict=True)
        )
        for record_id, _, _, metadata in records:
            current = existing_metadata.get(record_id)
            if current is not None and (
                current.get("document_id") != metadata["document_id"]
                or current.get("chunk_index") != metadata["chunk_index"]
            ):
                raise VectorStoreError(
                    f"Existing chunk ID conflicts with chunk metadata: {record_id}"
                )

        self._collection.upsert(
            ids=ids,
            documents=[record[1] for record in records],
            embeddings=[record[2].tolist() for record in records],
            metadatas=[record[3] for record in records],
        )
        return ids

    def get_by_id(self, chunk_id_value: str) -> StoredChunk | None:
        """Retrieve one stored chunk by its unique ID."""

        if not chunk_id_value.strip():
            raise ValueError("chunk_id must not be empty")

        result = self._collection.get(
            ids=[chunk_id_value],
            include=["documents", "embeddings", "metadatas"],
        )
        if not result["ids"]:
            return None

        metadata = result["metadatas"][0]
        if metadata is None:
            raise VectorStoreError(
                f"Stored chunk has no metadata: {chunk_id_value}"
            )
        embedding = np.asarray(result["embeddings"][0], dtype=np.float32)
        return StoredChunk(
            chunk_id=result["ids"][0],
            text=result["documents"][0],
            embedding=embedding,
            document_id=str(metadata["document_id"]),
            page_numbers=_parse_page_numbers(str(metadata["page_numbers"])),
            chunk_index=int(metadata["chunk_index"]),
            token_count=int(metadata["token_count"]),
        )

    @staticmethod
    def _validated_record(
        embedded_chunk: EmbeddedChunk,
    ) -> tuple[str, str, np.ndarray, dict[str, Any]]:
        chunk = embedded_chunk.chunk
        if not isinstance(chunk.document_id, str) or not chunk.document_id.strip():
            raise VectorStoreError("Chunk document_id must not be empty")
        if not chunk.text.strip():
            raise VectorStoreError("Chunk text must not be empty")
        if not chunk.page_numbers or any(
            isinstance(page, bool) or not isinstance(page, int) or page <= 0
            for page in chunk.page_numbers
        ):
            raise VectorStoreError("Chunk page_numbers must be positive integers")
        if (
            isinstance(chunk.chunk_index, bool)
            or not isinstance(chunk.chunk_index, int)
            or chunk.chunk_index < 0
        ):
            raise VectorStoreError("Chunk chunk_index must be a non-negative integer")
        if (
            isinstance(chunk.token_count, bool)
            or not isinstance(chunk.token_count, int)
            or chunk.token_count < 0
        ):
            raise VectorStoreError("Chunk token_count must be a non-negative integer")

        try:
            embedding = np.asarray(embedded_chunk.embedding, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise VectorStoreError("Chunk embedding must be numeric") from error
        if embedding.ndim != 1 or embedding.shape[0] != EXPECTED_EMBEDDING_DIMENSION:
            raise VectorStoreError(
                f"Expected embedding dimension {EXPECTED_EMBEDDING_DIMENSION}, "
                f"received shape {embedding.shape}"
            )

        metadata = {
            "document_id": chunk.document_id,
            "page_numbers": ",".join(str(page) for page in chunk.page_numbers),
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
        }
        return chunk_id(embedded_chunk), chunk.text, embedding, metadata


def _parse_page_numbers(value: str) -> list[int]:
    try:
        pages = [int(page) for page in value.split(",") if page]
    except ValueError as error:
        raise VectorStoreError("Stored page_numbers metadata is invalid") from error
    if not pages:
        raise VectorStoreError("Stored page_numbers metadata is empty")
    return pages
