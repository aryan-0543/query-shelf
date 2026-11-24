"""Manually verify generated-answer processing and source validation."""

from app.context_builder import ContextBuilder, ContextBuilderError
from app.llm_client import LLMClient, LLMClientError
from app.prompt_builder import PromptBuilder, PromptBuilderError
from app.query import parse_query
from app.response_processor import ResponseProcessorError, process_response
from app.retriever import Retriever, RetrieverError


def main() -> None:
    """Run retrieval, generation, and response processing for a sample query."""

    try:
        query = parse_query("What principles define clean code?")
        retrieval = Retriever().retrieve(query)
        context = ContextBuilder().build(retrieval)
        prompt = PromptBuilder().build(query, context)
        llm_response = LLMClient().generate(prompt)
        answer = process_response(llm_response, retrieval)
    except (
        ContextBuilderError,
        LLMClientError,
        PromptBuilderError,
        ResponseProcessorError,
        RetrieverError,
        ValueError,
    ) as error:
        print(f"Response processing failed: {error}")
        return

    print(f"Model: {answer.model}")
    print(f"Generation time: {answer.generation_seconds:.2f}s")
    print("Answer:")
    print(answer.answer)
    print("Verified sources:")
    if not answer.sources:
        print("  None")
    for source in answer.sources:
        print(f"  document_id={source.document_id}, page={source.page_number}")


if __name__ == "__main__":
    main()
