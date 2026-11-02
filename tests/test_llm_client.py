from types import SimpleNamespace

import pytest

from app.llm_client import LLMClient, LLMClientError
from app.context_builder import ContextResult
from app.prompt_builder import PromptBuilder
from app.query import UserQuery


def prompt(tmp_path):
    system = tmp_path / "system.md"
    system.write_text("System rules", encoding="utf-8")
    return PromptBuilder(system_prompt_path=system).build(
        UserQuery("Question"),
        ContextResult((), (), "Context", 1, 10),
    )


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, models):
        self.models = models


def test_generate_sends_separate_system_and_user_content(tmp_path) -> None:
    models = FakeModels(SimpleNamespace(text=" answer "))
    client = LLMClient(
        api_key="key",
        model="model",
        client=FakeClient(models),
    )

    result = client.generate(prompt(tmp_path))

    assert result.text == "answer"
    assert result.model == "model"
    call = models.calls[0]
    assert "Context" in call["contents"]
    assert call["config"].system_instruction == "System rules"


@pytest.mark.parametrize("name", ["LLM_API_KEY", "LLM_MODEL"])
def test_missing_configuration_is_rejected(monkeypatch, name) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    values = {"LLM_API_KEY": "key", "LLM_MODEL": "model"}
    values[name] = ""

    with pytest.raises(LLMClientError, match=name):
        LLMClient(api_key=values["LLM_API_KEY"], model=values["LLM_MODEL"])


def test_provider_failure_is_chained(tmp_path) -> None:
    error = RuntimeError("provider failed")
    client = LLMClient(
        api_key="key",
        model="model",
        client=FakeClient(FakeModels(error=error)),
    )

    with pytest.raises(LLMClientError) as raised:
        client.generate(prompt(tmp_path))

    assert raised.value.__cause__ is error


@pytest.mark.parametrize("response", [SimpleNamespace(text=""), SimpleNamespace(text=None)])
def test_empty_response_is_rejected(tmp_path, response) -> None:
    client = LLMClient(
        api_key="key",
        model="model",
        client=FakeClient(FakeModels(response=response)),
    )

    with pytest.raises(LLMClientError, match="empty"):
        client.generate(prompt(tmp_path))


def test_invalid_prompt_is_rejected() -> None:
    client = LLMClient(api_key="key", model="model", client=FakeClient(FakeModels()))

    with pytest.raises(LLMClientError, match="Prompt"):
        client.generate("invalid")  # type: ignore[arg-type]
