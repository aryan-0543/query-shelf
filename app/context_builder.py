"""Build bounded, source-traceable context from retrieval results."""

from collections.abc import Callable
from dataclasses import dataclass

from app.retriever import RetrievedChunk, RetrievalResult

DEFAULT_MAX_CONTEXT_TOKENS = 2_000
CONTEXT_SEPARATOR = "\n\n"


class ContextBuilderError(Exception):
    """Raised when retrieved context cannot be prepared."""


@dataclass(frozen=True)
class ContextResult:
    """Immutable context prepared from retrieved chunks."""

    included_chunks: tuple[RetrievedChunk, ...]
    excluded_chunks: tuple[RetrievedChunk, ...]
    formatted_context: str
    token_count: int
    max_context_tokens: int

    @property
    def included_count(self) -> int:
        """Return the number of chunks included in the context."""

        return len(self.included_chunks)

    @property
    def excluded_count(self) -> int:
        """Return the number of chunks excluded by the budget."""

        return len(self.excluded_chunks)


class ContextBuilder:
    """Format retrieved chunks while enforcing a context token budget."""

    def __init__(
        self,
        *,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if (
            isinstance(max_context_tokens, bool)
            or not isinstance(max_context_tokens, int)
            or max_context_tokens <= 0
        ):
            raise ContextBuilderError(
                "max_context_tokens must be a positive integer"
            )
        self._max_context_tokens = max_context_tokens
        self._token_counter = token_counter or _approximate_token_count

    def build(self, result: RetrievalResult) -> ContextResult:
        """Build ordered, source-labeled context from a retrieval result."""

        if not isinstance(result, RetrievalResult):
            raise ContextBuilderError("Input must be a RetrievalResult")

        chunk_ids = [chunk.chunk_id for chunk in result.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ContextBuilderError("Retrieval result contains duplicate chunk IDs")

        included: list[RetrievedChunk] = []
        excluded: list[RetrievedChunk] = []
        formatted_blocks: list[str] = []

        for chunk in result.chunks:
            block = _format_chunk(chunk)
            candidate_blocks = [*formatted_blocks, block]
            candidate_context = CONTEXT_SEPARATOR.join(candidate_blocks)
            candidate_tokens = self._count_tokens(candidate_context)
            if candidate_tokens <= self._max_context_tokens:
                included.append(chunk)
                formatted_blocks.append(block)
            else:
                excluded.append(chunk)

        formatted_context = CONTEXT_SEPARATOR.join(formatted_blocks)
        token_count = self._count_tokens(formatted_context)
        return ContextResult(
            included_chunks=tuple(included),
            excluded_chunks=tuple(excluded),
            formatted_context=formatted_context,
            token_count=token_count,
            max_context_tokens=self._max_context_tokens,
        )

    def _count_tokens(self, text: str) -> int:
        try:
            count = self._token_counter(text)
        except Exception as error:
            raise ContextBuilderError("Unable to count context tokens") from error
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ContextBuilderError(
                "Token counter must return a non-negative integer"
            )
        return count


def _format_chunk(chunk: RetrievedChunk) -> str:
    return (
        f"[Rank {chunk.rank} | Document: {chunk.document_id} | "
        f"Pages: {', '.join(str(page) for page in chunk.page_numbers)} | "
        f"Chunk: {chunk.chunk_id}]\n{chunk.text}"
    )


def _approximate_token_count(text: str) -> int:
    return len(text.split())
