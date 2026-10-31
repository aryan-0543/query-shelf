"""Validate and normalize user queries before retrieval processing."""

from dataclasses import dataclass

DEFAULT_MAX_QUERY_LENGTH = 2_000


class QueryValidationError(ValueError):
    """Raised when a user query is invalid."""


@dataclass(frozen=True)
class UserQuery:
    """A normalized natural-language user query."""

    text: str


def parse_query(
    value: str,
    *,
    max_length: int = DEFAULT_MAX_QUERY_LENGTH,
) -> UserQuery:
    """Validate and normalize a natural-language query."""

    if not isinstance(value, str):
        raise QueryValidationError("Query must be a string")
    if not isinstance(max_length, int) or isinstance(max_length, bool):
        raise QueryValidationError("max_length must be a positive integer")
    if max_length <= 0:
        raise QueryValidationError("max_length must be a positive integer")

    normalized = value.strip()
    if not normalized:
        raise QueryValidationError("Query must not be empty")
    if len(normalized) > max_length:
        raise QueryValidationError(
            f"Query exceeds the maximum length of {max_length} characters"
        )

    return UserQuery(text=normalized)
