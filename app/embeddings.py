"""Generate local embeddings for token-bounded text chunks."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.chunker import TextChunk
from app.model_config import EXPECTED_EMBEDDING_DIMENSION, MODEL_NAME
DEFAULT_BATCH_SIZE = 32


class EmbeddingError(Exception):
    """Raised when embeddings cannot be generated or validated."""


class EmbeddingInputError(EmbeddingError):
    """Raised when a chunk cannot be passed to the embedding model."""


@dataclass(frozen=True)
class EmbeddedChunk:
    """A source chunk paired with its generated embedding."""

    chunk: TextChunk
    embedding: np.ndarray


def load_embedding_model() -> Any:
    """Load and return the local sentence-transformers model."""

    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME)
    except Exception as error:
        raise EmbeddingError(
            f"Unable to load embedding model {MODEL_NAME!r}"
        ) from error


def _validate_inputs(
    chunks: list[TextChunk],
    tokenizer: Any,
    max_seq_length: int,
) -> None:
    for chunk in chunks:
        try:
            token_ids = tokenizer.encode(
                chunk.text,
                add_special_tokens=True,
                truncation=False,
            )
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            raise EmbeddingInputError(
                f"Unable to tokenize chunk {chunk.chunk_index}"
            ) from error

        if len(token_ids) > max_seq_length:
            raise EmbeddingInputError(
                f"Chunk {chunk.chunk_index} has {len(token_ids)} tokens, "
                f"but the model maximum is {max_seq_length}"
            )


def embed_chunks(
    chunks: list[TextChunk],
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[EmbeddedChunk]:
    """Batch-encode chunks and preserve each chunk beside its vector."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not chunks:
        return []

    model = model or load_embedding_model()
    tokenizer = tokenizer or model.tokenizer
    max_seq_length = getattr(model, "max_seq_length", None)
    if not isinstance(max_seq_length, int) or max_seq_length <= 0:
        raise EmbeddingError("Embedding model has no valid max_seq_length")

    _validate_inputs(chunks, tokenizer, max_seq_length)

    try:
        vectors = np.asarray(
            model.encode(
                [chunk.text for chunk in chunks],
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(chunks) > batch_size,
            )
        )
    except Exception as error:
        raise EmbeddingError("Unable to encode text chunks") from error

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise EmbeddingError(
            f"Expected one embedding per chunk, received shape {vectors.shape}"
        )
    if vectors.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise EmbeddingError(
            f"Expected embedding dimension {EXPECTED_EMBEDDING_DIMENSION}, "
            f"received {vectors.shape[1]}"
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingError("Embedding values must be finite")

    return [
        EmbeddedChunk(chunk=chunk, embedding=vector)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
