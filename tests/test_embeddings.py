import numpy as np
import pytest

from app.chunker import TextChunk
from app.embeddings import (
    DEFAULT_BATCH_SIZE,
    EXPECTED_EMBEDDING_DIMENSION,
    EmbeddingError,
    EmbeddingInputError,
    embed_chunks,
)


class FakeTokenizer:
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> list[int]:
        tokens = text.split()
        return ([101] if add_special_tokens else []) + list(range(len(tokens))) + (
            [102] if add_special_tokens else []
        )


class FakeModel:
    max_seq_length = 8

    def __init__(self, output: np.ndarray | None = None) -> None:
        self.tokenizer = FakeTokenizer()
        self.output = output
        self.calls: list[dict[str, object]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append({"texts": texts, **kwargs})
        if self.output is not None:
            return self.output
        return np.ones((len(texts), EXPECTED_EMBEDDING_DIMENSION))


def chunks() -> list[TextChunk]:
    return [
        TextChunk("document", (2,), 0, "first chunk", 4),
        TextChunk("document", (3,), 1, "second chunk", 4),
    ]


def test_embeddings_are_batched_and_metadata_is_preserved() -> None:
    model = FakeModel()
    input_chunks = chunks()

    result = embed_chunks(input_chunks, model=model, batch_size=DEFAULT_BATCH_SIZE)

    assert len(result) == 2
    assert result[0].chunk is input_chunks[0]
    assert result[0].chunk.chunk_index == 0
    assert result[1].chunk.page_numbers == (3,)
    assert result[0].embedding.shape == (EXPECTED_EMBEDDING_DIMENSION,)
    assert model.calls[0]["batch_size"] == DEFAULT_BATCH_SIZE
    assert model.calls[0]["show_progress_bar"] is False


def test_progress_is_enabled_for_large_batches() -> None:
    model = FakeModel()
    input_chunks = [
        TextChunk("document", (1,), index, "chunk", 3)
        for index in range(DEFAULT_BATCH_SIZE + 1)
    ]

    embed_chunks(input_chunks, model=model)

    assert model.calls[0]["show_progress_bar"] is True


def test_empty_input_does_not_load_or_encode() -> None:
    assert embed_chunks([], model=None) == []


def test_invalid_token_length_is_rejected() -> None:
    long_chunk = TextChunk("document", (1,), 0, " ".join(["word"] * 8), 10)

    with pytest.raises(EmbeddingInputError, match="maximum is 8"):
        embed_chunks([long_chunk], model=FakeModel())


def test_invalid_embedding_dimension_is_rejected() -> None:
    model = FakeModel(output=np.ones((1, 10)))

    with pytest.raises(EmbeddingError, match="embedding dimension"):
        embed_chunks(chunks()[:1], model=model)


def test_invalid_embedding_shape_is_rejected() -> None:
    model = FakeModel(output=np.ones(EXPECTED_EMBEDDING_DIMENSION))

    with pytest.raises(EmbeddingError, match="one embedding per chunk"):
        embed_chunks(chunks()[:1], model=model)


def test_encoding_errors_are_wrapped() -> None:
    class FailingModel(FakeModel):
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            raise RuntimeError("encoding failed")

    with pytest.raises(EmbeddingError, match="Unable to encode"):
        embed_chunks(chunks()[:1], model=FailingModel())


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_embeddings_are_rejected(value: float) -> None:
    output = np.ones((1, EXPECTED_EMBEDDING_DIMENSION))
    output[0, 0] = value

    with pytest.raises(EmbeddingError, match="finite"):
        embed_chunks(chunks()[:1], model=FakeModel(output=output))
