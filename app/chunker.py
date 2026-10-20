"""Create token-bounded chunks from extracted PDF page text."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer

from app.pdf_extractor import PageText

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 202
CHUNK_OVERLAP = 42
MODEL_MAX_LENGTH = 256


class ChunkingError(Exception):
    """Raised when page text cannot be tokenized or chunked."""


@dataclass(frozen=True)
class TextChunk:
    """A token-bounded chunk with source metadata."""

    document_id: str
    page_numbers: tuple[int, ...]
    chunk_index: int
    text: str
    token_count: int


def _load_tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def _split_text_units(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text)]
    units: list[str] = []

    for paragraph in paragraphs:
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        units.extend(sentence.strip() for sentence in sentences if sentence.strip())

    return units


def _tokenize(tokenizer: Any, text: str, *, special_tokens: bool) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=special_tokens)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ChunkingError("Unable to tokenize extracted PDF text") from error


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    """Tokenize raw text without constructing an oversized input sequence."""

    try:
        if hasattr(tokenizer, "tokenize") and hasattr(tokenizer, "convert_tokens_to_ids"):
            tokens: list[str] = []
            for word in text.split():
                tokens.extend(tokenizer.tokenize(word))
            return tokenizer.convert_tokens_to_ids(tokens)
        return tokenizer.encode(text, add_special_tokens=False)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ChunkingError("Unable to tokenize extracted PDF text") from error


def _text_chunks_for_page(
    page: PageText,
    tokenizer: Any,
    text_token_limit: int,
    overlap: int,
) -> list[tuple[str, tuple[int, ...]]]:
    units = _split_text_units(page.text)
    if not units:
        return []

    token_ids = [
        token_id
        for unit in units
        for token_id in _tokenize_text(tokenizer, unit)
    ]
    chunks: list[tuple[str, tuple[int, ...]]] = []
    start = 0

    while start < len(token_ids):
        end = min(start + text_token_limit, len(token_ids))
        safe_end = end

        while safe_end > start:
            text = tokenizer.decode(
                token_ids[start:safe_end],
                skip_special_tokens=True,
            ).strip()
            if len(_tokenize(tokenizer, text, special_tokens=True)) <= (
                text_token_limit + tokenizer.num_special_tokens_to_add(pair=False)
            ):
                chunks.append((text, (page.page_number,)))
                break
            safe_end -= 1
        else:
            raise ChunkingError(
                f"Unable to create a token-bounded chunk for page {page.page_number}"
            )

        if safe_end == len(token_ids):
            break

        next_start = safe_end - overlap
        if next_start <= start:
            raise ChunkingError(
                f"Chunk overlap prevents progress on page {page.page_number}"
            )
        start = next_start

    return chunks


def chunk_pages(
    pages: Iterable[PageText],
    document_id: str,
    *,
    tokenizer: Any | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Create token-bounded chunks from extracted pages.

    ``chunk_size`` is the complete tokenizer input budget, including special
    tokens. The text portion is therefore reduced by the tokenizer's special
    token count. Chunk overlap is measured in text tokens.
    """

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")

    tokenizer = tokenizer or _load_tokenizer()
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    text_token_limit = chunk_size - special_token_count
    if text_token_limit <= overlap:
        raise ValueError("chunk_size must leave room beyond the overlap")
    if chunk_size > MODEL_MAX_LENGTH:
        raise ValueError(f"chunk_size cannot exceed {MODEL_MAX_LENGTH} tokens")

    chunks: list[TextChunk] = []
    for page in pages:
        for text, page_numbers in _text_chunks_for_page(
            page, tokenizer, text_token_limit, overlap
        ):
            encoded_count = len(_tokenize(tokenizer, text, special_tokens=True))
            if encoded_count > chunk_size:
                raise ChunkingError(
                    f"Chunk for page {page.page_number} exceeds the token limit"
                )
            chunks.append(
                TextChunk(
                    document_id=document_id,
                    page_numbers=page_numbers,
                    chunk_index=len(chunks),
                    text=text,
                    token_count=encoded_count,
                )
            )

    return chunks
