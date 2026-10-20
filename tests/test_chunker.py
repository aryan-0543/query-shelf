from app.chunker import CHUNK_OVERLAP, CHUNK_SIZE, TextChunk, chunk_pages
from app.pdf_extractor import PageText


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.split()
        return ([101] if add_special_tokens else []) + list(range(len(tokens))) + (
            [102] if add_special_tokens else []
        )

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"token-{token}" for token in tokens if token not in {101, 102})

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return list(range(len(tokens)))


class GuardedTokenizer(FakeTokenizer):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if len(text.split()) > 512:
            raise AssertionError("oversized text passed to encode")
        return super().encode(text, add_special_tokens=add_special_tokens)


class ExpandingDecodeTokenizer(FakeTokenizer):
    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        decoded = super().decode(tokens, skip_special_tokens=skip_special_tokens)
        if tokens and tokens[0] >= 158:
            decoded += " expansion expansion expansion"
        return decoded


def test_chunk_size_includes_special_tokens() -> None:
    pages = [PageText(page_number=1, text=" ".join(["word"] * 250))]

    chunks = chunk_pages(pages, "document", tokenizer=FakeTokenizer())

    assert chunks
    assert all(chunk.token_count <= CHUNK_SIZE for chunk in chunks)
    assert all(chunk.token_count <= 256 for chunk in chunks)


def test_chunks_have_text_token_overlap() -> None:
    pages = [PageText(page_number=1, text=" ".join(["word"] * 350))]

    chunks = chunk_pages(pages, "document", tokenizer=FakeTokenizer())

    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()
    assert first_tokens[-CHUNK_OVERLAP:] == second_tokens[:CHUNK_OVERLAP]


def test_chunk_metadata_is_preserved() -> None:
    pages = [PageText(page_number=7, text="A short paragraph.")]

    assert chunk_pages(pages, "manual-document", tokenizer=FakeTokenizer()) == [
        TextChunk(
            document_id="manual-document",
            page_numbers=(7,),
            chunk_index=0,
            text="token-0 token-1 token-2",
            token_count=5,
        )
    ]


def test_empty_pages_and_empty_input_produce_no_chunks() -> None:
    tokenizer = FakeTokenizer()

    assert chunk_pages([], "document", tokenizer=tokenizer) == []
    assert (
        chunk_pages([PageText(page_number=1, text=" \n\n ")], "document", tokenizer=tokenizer)
        == []
    )


def test_paragraphs_are_packed_before_splitting() -> None:
    pages = [
        PageText(
            page_number=2,
            text="First sentence.\n\nSecond sentence.",
        )
    ]

    chunks = chunk_pages(pages, "document", tokenizer=FakeTokenizer())

    assert len(chunks) == 1
    assert chunks[0].page_numbers == (2,)


def test_overlap_does_not_make_a_valid_unit_exceed_the_limit() -> None:
    pages = [
        PageText(
            page_number=3,
            text=f"{'word ' * 190}. {'word ' * 30}.",
        )
    ]

    chunks = chunk_pages(pages, "document", tokenizer=FakeTokenizer())

    assert len(chunks) == 2
    assert all(chunk.token_count <= CHUNK_SIZE for chunk in chunks)


def test_long_page_is_windowed_before_tokenizer_encode() -> None:
    pages = [PageText(page_number=8, text=" ".join(["word"] * 600))]

    chunks = chunk_pages(pages, "document", tokenizer=GuardedTokenizer())

    assert len(chunks) > 3
    assert all(chunk.token_count <= CHUNK_SIZE for chunk in chunks)


def test_toc_like_long_page_handles_decode_boundary_expansion() -> None:
    toc_text = "\n".join(
        f"Chapter {index}: Meaningful title ........................ {index}"
        for index in range(120)
    )

    chunks = chunk_pages(
        [PageText(page_number=8, text=toc_text)],
        "sample-document",
        tokenizer=ExpandingDecodeTokenizer(),
    )

    assert len(chunks) > 1
    assert all(chunk.token_count <= CHUNK_SIZE for chunk in chunks)
    assert all(chunk.page_numbers == (8,) for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
