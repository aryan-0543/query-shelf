from dataclasses import FrozenInstanceError

import pytest

from app.llm_client import LLMResponse
from app.query import UserQuery
from app.response_processor import (
    GeneratedAnswer,
    GeneratedSource,
    ResponseProcessorError,
    process_response,
)
from app.retriever import RetrievedChunk, RetrievalResult


def chunk(
    document_id: str = "doc-a",
    pages: tuple[int, ...] = (2, 3, 5),
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{document_id}-chunk",
        text="retrieved text",
        document_id=document_id,
        page_numbers=pages,
        chunk_index=0,
        token_count=4,
        distance=0.1,
        rank=1,
    )


def retrieval(*chunks: RetrievedChunk) -> RetrievalResult:
    return RetrievalResult(UserQuery("question"), tuple(chunks), 0.1, 0.2)


def response(text: str = " Answer [p. 3] ", model: str = "model") -> LLMResponse:
    return LLMResponse(text=text, model=model, generation_seconds=1.25)


def test_processes_answer_and_verifies_page_source() -> None:
    result = process_response(response(), retrieval(chunk()))

    assert result == GeneratedAnswer(
        answer="Answer [p. 3]",
        model="model",
        generation_seconds=1.25,
        sources=(GeneratedSource("doc-a", 3),),
    )


def test_recognizes_ranges_lists_case_insensitively_and_deduplicates() -> None:
    answer = "Pages 2-3, 5; PP. 3-5; p. 2; page 5."

    result = process_response(response(answer), retrieval(chunk()))

    assert result.sources == tuple(
        GeneratedSource("doc-a", page) for page in (2, 3, 5)
    )
    assert result.answer == answer


def test_sources_preserve_verified_document_page_combinations() -> None:
    answer = "See pages 2-3."

    result = process_response(
        response(answer),
        retrieval(chunk("doc-a", (2,)), chunk("doc-b", (2, 3))),
    )

    assert result.sources == (
        GeneratedSource("doc-a", 2),
        GeneratedSource("doc-b", 2),
        GeneratedSource("doc-b", 3),
    )


def test_unverified_citations_are_not_sources_and_answer_is_preserved() -> None:
    answer = "Unsupported claim [page 99], but page 3 is available."

    result = process_response(response(answer), retrieval(chunk()))

    assert result.answer == answer
    assert result.sources == (GeneratedSource("doc-a", 3),)


@pytest.mark.parametrize("text", ["", "  \n  ", None])
def test_empty_or_malformed_response_text_is_rejected(text) -> None:
    invalid = LLMResponse(text=text, model="model", generation_seconds=1.0)

    with pytest.raises(ResponseProcessorError, match="text"):
        process_response(invalid, retrieval(chunk()))


@pytest.mark.parametrize(
    ("llm_response", "retrieval_result"),
    [
        ("not a response", retrieval(chunk())),
        (response(), "not a retrieval result"),
    ],
)
def test_malformed_inputs_are_rejected(llm_response, retrieval_result) -> None:
    with pytest.raises(ResponseProcessorError):
        process_response(llm_response, retrieval_result)


@pytest.mark.parametrize(
    "bad_chunk",
    [
        RetrievedChunk("id", "text", "", (2,), 0, 1, 0.1, 1),
        RetrievedChunk("id", "text", "doc", (), 0, 1, 0.1, 1),
        RetrievedChunk("id", "text", "doc", (0,), 0, 1, 0.1, 1),
    ],
)
def test_missing_or_invalid_source_metadata_is_rejected(bad_chunk) -> None:
    with pytest.raises(ResponseProcessorError, match="metadata"):
        process_response(response(), retrieval(bad_chunk))


def test_missing_retrieved_chunk_is_rejected() -> None:
    with pytest.raises(ResponseProcessorError, match="metadata"):
        process_response(response(), retrieval(None))  # type: ignore[arg-type]


def test_answer_structures_are_immutable() -> None:
    result = process_response(response(), retrieval(chunk()))

    with pytest.raises(FrozenInstanceError):
        result.answer = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.sources[0].page_number = 9  # type: ignore[misc]
    assert isinstance(result.sources, tuple)
