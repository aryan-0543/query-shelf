"""Normalize generated answers and verify cited source pages."""

import math
import re
from dataclasses import dataclass

from app.llm_client import LLMResponse
from app.retriever import RetrievedChunk, RetrievalResult

_PAGE_CITATION = re.compile(
    r"\b(?:pages?|pp?\.?)\s+"
    r"((?:\d+(?:\s*-\s*\d+)?)(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*)",
    re.IGNORECASE,
)
_PAGE_ITEM = re.compile(r"(\d+)(?:\s*-\s*(\d+))?")


class ResponseProcessorError(Exception):
    """Raised when an LLM response or retrieval metadata is invalid."""


@dataclass(frozen=True)
class GeneratedSource:
    """A cited document page verified against retrieved chunks."""

    document_id: str
    page_number: int


@dataclass(frozen=True)
class GeneratedAnswer:
    """An immutable generated answer with verified page sources."""

    answer: str
    model: str
    generation_seconds: float
    sources: tuple[GeneratedSource, ...]


def process_response(
    response: LLMResponse,
    retrieval: RetrievalResult,
) -> GeneratedAnswer:
    """Normalize answer whitespace and return only verified citations."""

    if not isinstance(response, LLMResponse):
        raise ResponseProcessorError("Response must be an LLMResponse")
    if not isinstance(retrieval, RetrievalResult):
        raise ResponseProcessorError("Retrieval must be a RetrievalResult")

    answer = response.text.strip() if isinstance(response.text, str) else ""
    if not answer:
        raise ResponseProcessorError("LLM response text must not be empty")
    if not isinstance(response.model, str) or not response.model.strip():
        raise ResponseProcessorError("LLM response model must not be empty")
    if (
        isinstance(response.generation_seconds, bool)
        or not isinstance(response.generation_seconds, (int, float))
        or not math.isfinite(response.generation_seconds)
        or response.generation_seconds < 0
    ):
        raise ResponseProcessorError("LLM generation time is invalid")
    if not isinstance(retrieval.chunks, tuple):
        raise ResponseProcessorError("Retrieved chunk metadata is malformed")

    pages_by_document: dict[str, set[int]] = {}
    for chunk in retrieval.chunks:
        if not isinstance(chunk, RetrievedChunk):
            raise ResponseProcessorError("Retrieved chunk metadata is malformed")
        if not isinstance(chunk.document_id, str) or not chunk.document_id.strip():
            raise ResponseProcessorError(
                "Retrieved chunk document_id metadata is missing or invalid"
            )
        if (
            not isinstance(chunk.page_numbers, tuple)
            or not chunk.page_numbers
            or any(
                isinstance(page, bool) or not isinstance(page, int) or page <= 0
                for page in chunk.page_numbers
            )
        ):
            raise ResponseProcessorError(
                "Retrieved chunk page_numbers metadata is missing or invalid"
            )
        pages_by_document.setdefault(chunk.document_id, set()).update(
            chunk.page_numbers
        )

    sources: list[GeneratedSource] = []
    seen_sources: set[tuple[str, int]] = set()
    documents_by_page: dict[int, list[str]] = {}
    for document_id, pages in pages_by_document.items():
        for page in pages:
            documents_by_page.setdefault(page, []).append(document_id)
    available_pages = sorted(documents_by_page)

    for citation in _PAGE_CITATION.finditer(answer):
        for item in _PAGE_ITEM.finditer(citation.group(1)):
            start = int(item.group(1))
            end = int(item.group(2) or start)
            if end < start:
                continue
            for page in available_pages:
                if start <= page <= end:
                    for document_id in documents_by_page[page]:
                        source_key = (document_id, page)
                        if source_key not in seen_sources:
                            seen_sources.add(source_key)
                            sources.append(GeneratedSource(document_id, page))

    return GeneratedAnswer(
        answer=answer,
        model=response.model.strip(),
        generation_seconds=float(response.generation_seconds),
        sources=tuple(sources),
    )
