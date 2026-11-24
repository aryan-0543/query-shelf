from dataclasses import FrozenInstanceError

import pytest

from app.context_builder import ContextResult
from app.llm_client import LLMResponse
from app.pipeline import PipelineTimings, RAGPipeline, RAGPipelineError
from app.query import UserQuery
from app.response_processor import GeneratedAnswer, GeneratedSource
from app.retriever import RetrievedChunk, RetrievalResult


def retrieval(chunks: tuple[RetrievedChunk, ...]) -> RetrievalResult:
    return RetrievalResult(UserQuery("question"), chunks, 0.1, 0.2)


def retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk("chunk", "text", "document", (3,), 0, 1, 0.1, 1)


class FakeRetriever:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def retrieve(self, query):
        self.calls.append(query)
        return self.result


class FakeContextBuilder:
    def __init__(self):
        self.calls = []
        self.result = None

    def build(self, result):
        self.calls.append(result)
        self.result = ContextResult(result.chunks, (), "context", 1, 100)
        return self.result


class FakePromptBuilder:
    def __init__(self):
        self.calls = []

    def build(self, query, context):
        self.calls.append((query, context))
        return "prompt"


class FakeLLMClient:
    def __init__(self, response=None):
        self.response = response or LLMResponse("answer", "model", 0.4)
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.response


def test_pipeline_runs_components_in_order_and_returns_answer() -> None:
    query = UserQuery("question")
    retrieved = retrieval((retrieved_chunk(),))
    retriever = FakeRetriever(retrieved)
    context_builder = FakeContextBuilder()
    prompt_builder = FakePromptBuilder()
    llm_client = FakeLLMClient()
    calls = []

    def process(response, retrieval_result):
        calls.append((response, retrieval_result))
        return GeneratedAnswer(
            "answer",
            response.model,
            response.generation_seconds,
            (GeneratedSource("document", 3),),
        )

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
        response_processor=process,
    )

    answer = pipeline.run(query)

    assert answer.answer == "answer"
    assert retriever.calls == [query]
    assert context_builder.calls == [retrieved]
    assert prompt_builder.calls == [(query, context_builder.result)]
    assert llm_client.calls == ["prompt"]
    assert calls == [(llm_client.response, retrieved)]
    assert pipeline.last_timings is not None
    assert pipeline.last_timings.generation_seconds == 0.4
    assert pipeline.last_timings.total_seconds >= 0


def test_empty_retrieval_stops_before_other_components() -> None:
    context_builder = FakeContextBuilder()
    prompt_builder = FakePromptBuilder()
    llm_client = FakeLLMClient()
    pipeline = RAGPipeline(
        retriever=FakeRetriever(retrieval(())),
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )

    with pytest.raises(RAGPipelineError, match="no chunks"):
        pipeline.run(UserQuery("question"))

    assert context_builder.calls == []
    assert prompt_builder.calls == []
    assert llm_client.calls == []
    assert pipeline.last_timings is not None
    assert pipeline.last_timings.context_seconds == 0


def test_last_timings_reset_when_new_run_fails() -> None:
    retriever = FakeRetriever(retrieval((retrieved_chunk(),)))
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=FakeContextBuilder(),
        prompt_builder=FakePromptBuilder(),
        llm_client=FakeLLMClient(),
        response_processor=lambda response, result: GeneratedAnswer(
            "answer", "model", 0.1, ()
        ),
    )
    pipeline.run(UserQuery("first"))
    assert pipeline.last_timings is not None

    retriever.result = None

    with pytest.raises(RAGPipelineError):
        pipeline.run(UserQuery("second"))
    assert pipeline.last_timings is None


@pytest.mark.parametrize(
    ("stage", "component_name"),
    [
        ("retrieval", "retriever"),
        ("context building", "context_builder"),
        ("prompt building", "prompt_builder"),
        ("LLM generation", "llm_client"),
        ("response processing", "response_processor"),
    ],
)
def test_stage_failures_are_chained(stage, component_name) -> None:
    error = RuntimeError(f"{stage} broke")

    class FailingComponent:
        def retrieve(self, *args):
            raise error

        def build(self, *args):
            raise error

        def generate(self, *args):
            raise error

    def failing_processor(*args):
        raise error

    components = {
        "retriever": FakeRetriever(retrieval((retrieved_chunk(),))),
        "context_builder": FakeContextBuilder(),
        "prompt_builder": FakePromptBuilder(),
        "llm_client": FakeLLMClient(),
        "response_processor": lambda response, result: GeneratedAnswer(
            "answer", "model", 0.1, ()
        ),
    }
    components[component_name] = (
        failing_processor
        if component_name == "response_processor"
        else FailingComponent()
    )
    pipeline = RAGPipeline(
        retriever=components["retriever"],
        context_builder=components["context_builder"],
        prompt_builder=components["prompt_builder"],
        llm_client=components["llm_client"],
        response_processor=components["response_processor"],
    )

    with pytest.raises(RAGPipelineError, match=stage) as raised:
        pipeline.run(UserQuery("question"))

    assert raised.value.__cause__ is error


def test_pipeline_timings_are_immutable() -> None:
    timings = PipelineTimings(1, 0.2, 0.1, 0.1, 0.4, 0.2)
    with pytest.raises(FrozenInstanceError):
        timings.total_seconds = 3  # type: ignore[misc]
