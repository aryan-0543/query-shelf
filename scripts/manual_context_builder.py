"""Manually verify context construction from persistent retrieval results."""

from app.context_builder import ContextBuilder, ContextBuilderError
from app.query import parse_query
from app.retriever import Retriever, RetrieverError


def main() -> None:
    """Retrieve a sample query and display its bounded context."""

    try:
        query = parse_query("What principles define clean code?")
        retrieval = Retriever().retrieve(query)
        context = ContextBuilder().build(retrieval)
    except (ContextBuilderError, RetrieverError, ValueError) as error:
        print(f"Context building failed: {error}")
        return

    print(f"Query: {retrieval.query.text}")
    print(f"Included chunks: {context.included_count}")
    print(f"Excluded chunks: {context.excluded_count}")
    print(f"Total context tokens: {context.token_count}")
    print(f"Configured context budget: {context.max_context_tokens}")

    for chunk in context.included_chunks:
        print(
            f"Included rank {chunk.rank}: chunk_id={chunk.chunk_id}, "
            f"document_id={chunk.document_id}, pages={chunk.page_numbers}"
        )
    for chunk in context.excluded_chunks:
        print(
            f"Excluded rank {chunk.rank}: chunk_id={chunk.chunk_id}, "
            f"document_id={chunk.document_id}, pages={chunk.page_numbers}"
        )

    if not context.included_chunks:
        print("No chunks were included in the context.")
        return

    print("\nFormatted context:")
    print(context.formatted_context)


if __name__ == "__main__":
    main()
