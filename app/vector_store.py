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
    content_fingerprint: str | None = None


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

    def store_chunks(
        self,
        chunks: list[EmbeddedChunk],
        *,
        content_fingerprint: str | None = None,
    ) -> list[str]:
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
            metadatas=[
                {
                    **record[3],
                    **(
                        {"content_fingerprint": content_fingerprint}
                        if content_fingerprint is not None
                        else {}
                    ),
                }
                for record in records
            ],
        )
        return ids

    def replace_document(
        self,
        document_id: str,
        content_fingerprint: str,
        chunks: list[EmbeddedChunk],
    ) -> bool:
        """Replace a document's chunks when its content fingerprint changes.

        Returns ``True`` when storage changed and ``False`` when the existing
        document already has the same fingerprint.
        """

        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        if not content_fingerprint.strip():
            raise ValueError("content_fingerprint must not be empty")
        if any(item.chunk.document_id != document_id for item in chunks):
            raise VectorStoreError("All chunks must belong to document_id")

        existing = self._collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        existing_metadata = existing.get("metadatas", [])
        if existing_metadata and all(
            metadata
            and metadata.get("content_fingerprint") == content_fingerprint
            for metadata in existing_metadata
        ):
            return False

        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
        self.store_chunks(chunks, content_fingerprint=content_fingerprint)
        return True

    def is_current_document(
        self, document_id: str, content_fingerprint: str
    ) -> bool:
        """Return whether a stored document has the supplied fingerprint."""

        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        if not content_fingerprint.strip():
            raise ValueError("content_fingerprint must not be empty")
        existing = self._collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        metadata = existing.get("metadatas", [])
        return bool(metadata) and all(
            item and item.get("content_fingerprint") == content_fingerprint
            for item in metadata
        )

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
            content_fingerprint=metadata.get("content_fingerprint"),
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
        if not np.isfinite(embedding).all():
            raise VectorStoreError("Embedding values must be finite")

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
