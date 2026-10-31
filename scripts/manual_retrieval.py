"""Manually verify candidate retrieval."""

from app.query import parse_query
from app.retriever import Retriever, RetrieverError


def main() -> None:
    """Retrieve and display traceable candidate chunks."""

    query = parse_query("What principles define clean code?")
    try:
        result = Retriever().retrieve(query)
    except RetrieverError as error:
        print(f"Retrieval failed: {error}")
        return

    print(f"Query: {result.query.text}")
    print(f"Retrieved chunks: {result.total_chunks}")
    print(f"Embedding execution time: {result.query_embedding_seconds:.2f}s")
    print(f"Vector search execution time: {result.vector_search_seconds:.2f}s")
    for chunk in result.chunks:
        preview = " ".join(chunk.text.split())
        if len(preview) > 160:
            preview = f"{preview[:157]}..."
        print(
            f"Rank {chunk.rank}: chunk_id={chunk.chunk_id}, "
            f"document_id={chunk.document_id}, pages={chunk.page_numbers}, "
            f"distance={chunk.distance:.6f}"
        )
        print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
