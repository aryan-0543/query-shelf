"""Manually verify LLM generation with the configured provider credentials."""

from app.context_builder import ContextBuilder, ContextBuilderError
from app.llm_client import LLMClient, LLMClientError
from app.prompt_builder import PromptBuilder, PromptBuilderError
from app.query import parse_query
from app.retriever import Retriever, RetrieverError


def main() -> None:
    """Run the complete query-to-LLM flow without printing secrets."""

    try:
        query = parse_query("What principles define clean code?")
        retrieval = Retriever().retrieve(query)
        context = ContextBuilder().build(retrieval)
        prompt = PromptBuilder().build(query, context)
        response = LLMClient().generate(prompt)
    except (
        ContextBuilderError,
        LLMClientError,
        PromptBuilderError,
        RetrieverError,
        ValueError,
    ) as error:
        print(f"LLM generation failed: {error}")
        return

    print(f"Model: {response.model}")
    print("Response:")
    print(response.text)


if __name__ == "__main__":
    main()
