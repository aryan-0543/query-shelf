"""Manually exercise the complete RAG pipeline entry point."""

from app.pipeline import RAGPipeline, RAGPipelineError
from app.query import parse_query


def main() -> None:
    """Run one sample question through the complete pipeline."""

    try:
        query = parse_query("What principles define clean code?")
        pipeline = RAGPipeline()
        answer = pipeline.run(query)
    except (RAGPipelineError, ValueError) as error:
        print(f"RAG pipeline failed: {error}")
        return

    print(f"Question: {query.text}")
    print(f"Answer ({answer.model}):")
    print(answer.answer)
    print("Verified sources:")
    if not answer.sources:
        print("  None")
    for source in answer.sources:
        print(f"  {source.document_id}, page {source.page_number}")
    timings = pipeline.last_timings
    if timings is not None:
        print(
            "Timings: "
            f"total={timings.total_seconds:.2f}s, "
            f"retrieval={timings.retrieval_seconds:.2f}s, "
            f"context={timings.context_seconds:.2f}s, "
            f"prompt={timings.prompt_seconds:.2f}s, "
            f"generation={timings.generation_seconds:.2f}s, "
            f"response_processing={timings.response_processing_seconds:.2f}s"
        )


if __name__ == "__main__":
    main()
