from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.context_builder import ContextResult
from app.prompt_builder import PromptBuilder, PromptBuilderError
from app.query import UserQuery


def context(text: str = "[Rank 1 | Pages: 2]\nsource text") -> ContextResult:
    return ContextResult((), (), text, len(text.split()), 100)


def test_build_separates_system_and_user_content(tmp_path: Path) -> None:
    prompt_path = tmp_path / "system.md"
    prompt_path.write_text("Use context only.", encoding="utf-8")

    result = PromptBuilder(system_prompt_path=prompt_path).build(
        UserQuery("What is clean code?"),
        context(),
    )

    assert result.system_instructions == "Use context only."
    assert "<retrieved_context>" in result.user_message
    assert "source text" in result.user_message
    assert "<user_question>\nWhat is clean code?" in result.user_message
    assert result.query.text == "What is clean code?"
    assert result.context == context()


def test_empty_context_is_explicit() -> None:
    result = PromptBuilder().build(UserQuery("question"), context(""))

    assert "No retrieved context is available." in result.user_message


def test_system_prompt_file_is_required_and_non_empty(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    with pytest.raises(PromptBuilderError, match="Unable to load"):
        PromptBuilder(system_prompt_path=missing)

    empty = tmp_path / "empty.md"
    empty.write_text(" \n", encoding="utf-8")
    with pytest.raises(PromptBuilderError, match="must not be empty"):
        PromptBuilder(system_prompt_path=empty)


@pytest.mark.parametrize("invalid_query", ["query", None])
def test_invalid_query_is_rejected(invalid_query: object) -> None:
    builder = PromptBuilder()

    with pytest.raises(PromptBuilderError, match="Query"):
        builder.build(invalid_query, context())  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_context", ["context", None])
def test_invalid_context_is_rejected(invalid_context: object) -> None:
    builder = PromptBuilder()

    with pytest.raises(PromptBuilderError, match="Context"):
        builder.build(UserQuery("question"), invalid_context)  # type: ignore[arg-type]


def test_prompt_result_is_immutable() -> None:
    result = PromptBuilder().build(UserQuery("question"), context())

    with pytest.raises(FrozenInstanceError):
        result.user_message = "changed"  # type: ignore[misc]
