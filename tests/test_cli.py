from app.cli import run_cli
from app.pipeline import RAGPipelineError
from app.query import UserQuery
from app.response_processor import GeneratedAnswer, GeneratedSource


class FakePipeline:
    def __init__(self, answers=None, error=None) -> None:
        self.answers = list(answers or [])
        self.error = error
        self.queries: list[UserQuery] = []

    def run(self, query: UserQuery) -> GeneratedAnswer:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.answers.pop(0)


def answer(text: str = "Answer") -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=text,
        model="test-model",
        generation_seconds=0.25,
        sources=(GeneratedSource("document-1", 4),),
    )


def run_with_inputs(pipeline, values: list[str]):
    outputs: list[str] = []
    inputs = iter(values)
    run_cli(
        pipeline,
        input_fn=lambda prompt: next(inputs),
        output_fn=outputs.append,
    )
    return outputs


def test_cli_runs_multiple_questions_and_displays_answer_sources() -> None:
    pipeline = FakePipeline([answer("First answer"), answer("Second answer")])

    outputs = run_with_inputs(pipeline, [" first question ", "second question", "exit"])

    assert [query.text for query in pipeline.queries] == [
        "first question",
        "second question",
    ]
    joined = "\n".join(outputs)
    assert "First answer" in joined
    assert "Second answer" in joined
    assert "test-model" in joined
    assert "0.25s" in joined
    assert "document-1, page 4" in joined
    assert "chunk text" not in joined
    assert "Goodbye." in joined


def test_cli_reports_invalid_query_without_calling_pipeline() -> None:
    pipeline = FakePipeline([answer()])

    outputs = run_with_inputs(pipeline, ["   ", "exit"])

    assert pipeline.queries == []
    assert any("Invalid question" in output for output in outputs)


def test_cli_reports_pipeline_errors_and_continues() -> None:
    error = RAGPipelineError("Retrieval returned no chunks")
    pipeline = FakePipeline(error=error)

    outputs = run_with_inputs(pipeline, ["question", "quit"])

    assert any("Unable to answer question" in output for output in outputs)
    assert pipeline.queries == [UserQuery("question")]


def test_cli_reports_unexpected_error_concisely_and_continues() -> None:
    pipeline = FakePipeline(error=RuntimeError("private details"))

    outputs = run_with_inputs(pipeline, ["question", "exit"])

    joined = "\n".join(outputs)
    assert "Unexpected error while processing your question." in joined
    assert "private details" not in joined


def test_cli_handles_eof() -> None:
    outputs: list[str] = []

    def eof(prompt: str) -> str:
        raise EOFError

    run_cli(FakePipeline(), input_fn=eof, output_fn=outputs.append)

    assert outputs[-1] == "\nGoodbye."


def test_cli_handles_keyboard_interrupt() -> None:
    outputs: list[str] = []

    def interrupted(prompt: str) -> str:
        raise KeyboardInterrupt

    run_cli(FakePipeline(), input_fn=interrupted, output_fn=outputs.append)

    assert outputs[-1] == "\nGoodbye."


def test_cli_displays_no_sources_when_answer_has_none() -> None:
    pipeline = FakePipeline(
        [GeneratedAnswer("Answer", "test-model", 0.1, ())]
    )

    outputs = run_with_inputs(pipeline, ["question", "exit"])

    assert "Verified sources: none" in outputs
