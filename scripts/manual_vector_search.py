"""Manually verify native vector similarity search."""

from app.embeddings import EmbeddingError, embed_query
from app.query import parse_query
from app.vector_store import VectorStore, VectorStoreError


def main() -> None:
    """Search the persisted collection with a sample natural-language query."""

    try:
        query = parse_query("What principles define clean code?")
        query_embedding = embed_query(query)
        results = VectorStore().similarity_search(query_embedding)
    except (EmbeddingError, VectorStoreError, ValueError) as error:
        print(f"Vector search failed: {error}")
        return

    if not results:
        print("No matching chunks found.")
        return

    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.text.split())
        if len(preview) > 160:
            preview = f"{preview[:157]}..."
        print(
            f"Rank {rank}: chunk_id={result.chunk_id}, "
            f"document_id={result.document_id}, pages={result.page_numbers}, "
            f"distance={result.distance:.6f}"
        )
        print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
