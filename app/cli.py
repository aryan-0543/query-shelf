"""Interactive command-line interface for Query Shelf."""

from collections.abc import Callable

from app.llm_client import LLMClientError
from app.pipeline import RAGPipeline, RAGPipelineError
from app.query import QueryValidationError, parse_query
from app.response_processor import GeneratedAnswer


def run_cli(
    pipeline: RAGPipeline | None = None,
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> None:
    """Run an interactive question loop with injectable terminal functions."""

    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn
    active_pipeline = pipeline

    write_output("Welcome to Query Shelf")
    write_output("Ask a question about your indexed PDFs, or type 'exit' to quit.")

    while True:
        try:
            raw_query = read_input("You: ")
        except (EOFError, KeyboardInterrupt):
            write_output("\nGoodbye.")
            return

        command = raw_query.strip()
        if command.lower() in {"exit", "quit"}:
            write_output("Goodbye.")
            return

        try:
            query = parse_query(raw_query)
        except QueryValidationError as error:
            write_output(f"Invalid question: {error}")
            continue

        try:
            if active_pipeline is None:
                active_pipeline = RAGPipeline()
            answer = active_pipeline.run(query)
            _display_answer(answer, write_output)
        except RAGPipelineError as error:
            write_output(f"Unable to answer question: {error}")
        except LLMClientError as error:
            write_output(f"LLM configuration error: {error}")
        except Exception:
            write_output("Unexpected error while processing your question.")


def _display_answer(
    answer: GeneratedAnswer,
    output_fn: Callable[[str], None],
) -> None:
    output_fn(f"\nAnswer:\n{answer.answer}")
    output_fn(f"Model: {answer.model}")
    output_fn(f"Generation time: {answer.generation_seconds:.2f}s")
    if answer.sources:
        output_fn("Verified sources:")
        for source in answer.sources:
            output_fn(f"- {source.document_id}, page {source.page_number}")
    else:
        output_fn("Verified sources: none")
    output_fn("")


if __name__ == "__main__":
    run_cli()
