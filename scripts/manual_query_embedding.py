"""Manually verify embedding generation for one user query."""

from time import perf_counter

import numpy as np

from app.embeddings import EmbeddingError, embed_query
from app.query import UserQuery


def main() -> None:
    """Embed a sample query and report safe summary statistics."""

    query = UserQuery("What principles define clean code?")
    started = perf_counter()

    try:
        embedding = embed_query(query)
    except EmbeddingError as error:
        print(f"Query embedding failed: {error}")
        return

    elapsed = perf_counter() - started
    print(f"Embedding dimensions: {embedding.shape[0]}")
    print(f"Finite values: {bool(np.isfinite(embedding).all())}")
    print(f"Execution time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
