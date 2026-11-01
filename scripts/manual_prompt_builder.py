"""Manually verify grounded prompt construction."""

from app.context_builder import ContextBuilder, ContextBuilderError
from app.prompt_builder import PromptBuilder, PromptBuilderError
from app.query import parse_query
from app.retriever import Retriever, RetrieverError


def main() -> None:
    """Retrieve context and display the constructed prompt components."""

    try:
        query = parse_query("What principles define clean code?")
        retrieval = Retriever().retrieve(query)
        context = ContextBuilder().build(retrieval)
        prompt = PromptBuilder().build(query, context)
    except (ContextBuilderError, PromptBuilderError, RetrieverError, ValueError) as error:
        print(f"Prompt building failed: {error}")
        return

    print(f"Query: {prompt.query.text}")
    print(f"Included chunks: {context.included_count}")
    print(f"Excluded chunks: {context.excluded_count}")
    print("\nSystem instructions:")
    print(prompt.system_instructions)
    print("\nUser message:")
    print(prompt.user_message)


if __name__ == "__main__":
    main()
