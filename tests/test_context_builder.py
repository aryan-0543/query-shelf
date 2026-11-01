from dataclasses import FrozenInstanceError

import pytest

from app.context_builder import ContextBuilder, ContextBuilderError
from app.query import UserQuery
from app.retriever import RetrievedChunk, RetrievalResult


def retrieval_result(*chunks: RetrievedChunk) -> RetrievalResult:
    return RetrievalResult(
        query=UserQuery("What is clean code?"),
        chunks=tuple(chunks),
        query_embedding_seconds=0.1,
        vector_search_seconds=0.2,
    )


def chunk(
    chunk_id: str = "chunk-1",
    *,
    rank: int = 1,
    text: str = "retrieved text",
    document_id: str = "document",
    pages: tuple[int, ...] = (2,),
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        page_numbers=pages,
        chunk_index=rank - 1,
        token_count=len(text.split()),
        distance=0.2,
        rank=rank,
    )


def test_build_formats_context_and_preserves_metadata() -> None:
    first = chunk("first", rank=1, pages=(2, 3))
    second = chunk("second", rank=2, document_id="other")

    result = ContextBuilder().build(retrieval_result(first, second))

    assert result.included_chunks == (first, second)
    assert result.excluded_chunks == ()
    assert result.included_count == 2
    assert result.excluded_count == 0
    assert "[Rank 1 | Document: document | Pages: 2, 3 | Chunk: first]" in (
        result.formatted_context
    )
    assert result.formatted_context.index("first") < result.formatted_context.index(
        "second"
    )
    assert result.token_count <= result.max_context_tokens


def test_empty_retrieval_result_returns_empty_context() -> None:
    result = ContextBuilder().build(retrieval_result())

    assert result.included_chunks == ()
    assert result.excluded_chunks == ()
    assert result.formatted_context == ""
    assert result.token_count == 0


def test_duplicate_chunk_ids_are_rejected() -> None:
    duplicate = chunk()

    with pytest.raises(ContextBuilderError, match="duplicate"):
        ContextBuilder().build(retrieval_result(duplicate, duplicate))


def test_distinct_chunks_with_identical_text_are_retained() -> None:
    result = ContextBuilder().build(
        retrieval_result(chunk("first", text="same"), chunk("second", text="same", rank=2))
    )

    assert result.included_count == 2


def test_budget_excludes_chunks_without_truncating_and_continues() -> None:
    first = chunk("first", text="one two three")
    second = chunk("second", text="four")
    third = chunk("third", text="five", rank=3)

    def counter(text: str) -> int:
        return 100 if "four" in text else len(text.split())

    result = ContextBuilder(max_context_tokens=50, token_counter=counter).build(
        retrieval_result(first, second, third)
    )

    assert result.included_chunks == (first, third)
    assert result.excluded_chunks == (second,)
    assert "four" not in result.formatted_context
    assert "five" in result.formatted_context


def test_exact_budget_boundary_is_inclusive() -> None:
    first = chunk("first", text="one")
    counter = lambda text: len(text.split())
    expected_tokens = counter(
        "[Rank 1 | Document: document | Pages: 2 | Chunk: first]\none"
    )

    result = ContextBuilder(
        max_context_tokens=expected_tokens,
        token_counter=counter,
    ).build(retrieval_result(first))

    assert result.included_chunks == (first,)
    assert result.token_count == expected_tokens


def test_chunk_larger_than_budget_is_excluded() -> None:
    first = chunk("first", text="one two three")
    result = ContextBuilder(max_context_tokens=1).build(retrieval_result(first))

    assert result.included_chunks == ()
    assert result.excluded_chunks == (first,)
    assert result.formatted_context == ""


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "10"])
def test_invalid_context_budget_is_rejected(value: object) -> None:
    with pytest.raises(ContextBuilderError, match="max_context_tokens"):
        ContextBuilder(max_context_tokens=value)  # type: ignore[arg-type]


def test_invalid_input_is_rejected() -> None:
    with pytest.raises(ContextBuilderError, match="RetrievalResult"):
        ContextBuilder().build("invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_invalid_token_counter_result_is_rejected(value: object) -> None:
    with pytest.raises(ContextBuilderError, match="Token counter"):
        ContextBuilder(token_counter=lambda text: value).build(
            retrieval_result(chunk())
        )


def test_token_counter_failure_is_chained() -> None:
    error = RuntimeError("counter failed")

    def failing_counter(text: str) -> int:
        raise error

    with pytest.raises(ContextBuilderError) as raised:
        ContextBuilder(token_counter=failing_counter).build(
            retrieval_result(chunk())
        )

    assert raised.value.__cause__ is error


def test_context_result_is_immutable() -> None:
    result = ContextBuilder().build(retrieval_result(chunk()))

    with pytest.raises(FrozenInstanceError):
        result.formatted_context = "changed"  # type: ignore[misc]
    assert isinstance(result.included_chunks, tuple)
    assert isinstance(result.excluded_chunks, tuple)
