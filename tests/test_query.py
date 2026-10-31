import pytest

from app.query import (
    DEFAULT_MAX_QUERY_LENGTH,
    QueryValidationError,
    UserQuery,
    parse_query,
)


@pytest.mark.parametrize(
    "text",
    [
        "What is clean code?",
        "Explain the main topic of this paragraph.",
        "Give an example of the principle.",
        "Which page discusses chapters and sections?",
    ],
)
def test_valid_queries_are_returned_as_structured_objects(text: str) -> None:
    result = parse_query(text)

    assert result == UserQuery(text=text)


def test_query_trims_outer_whitespace_only() -> None:
    result = parse_query("  What   does this mean?  \n")

    assert result.text == "What   does this mean?"


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_empty_queries_are_rejected(value: str) -> None:
    with pytest.raises(QueryValidationError, match="must not be empty"):
        parse_query(value)


@pytest.mark.parametrize("value", [None, 42, ["question"]])
def test_non_string_queries_are_rejected(value: object) -> None:
    with pytest.raises(QueryValidationError, match="must be a string"):
        parse_query(value)  # type: ignore[arg-type]


def test_default_maximum_length_is_enforced() -> None:
    with pytest.raises(
        QueryValidationError,
        match=f"maximum length of {DEFAULT_MAX_QUERY_LENGTH}",
    ):
        parse_query("q" * (DEFAULT_MAX_QUERY_LENGTH + 1))


def test_custom_maximum_length_is_supported() -> None:
    assert parse_query("question", max_length=8).text == "question"

    with pytest.raises(QueryValidationError, match="maximum length of 7"):
        parse_query("question", max_length=7)


def test_invalid_maximum_length_is_rejected() -> None:
    with pytest.raises(QueryValidationError, match="positive integer"):
        parse_query("question", max_length=0)

    with pytest.raises(QueryValidationError, match="positive integer"):
        parse_query("question", max_length=True)


def test_query_object_is_immutable() -> None:
    query = parse_query("question")

    with pytest.raises(AttributeError):
        query.text = "changed"  # type: ignore[misc]
