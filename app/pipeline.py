"""Coordinate the complete retrieval-augmented generation workflow."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from app.context_builder import ContextBuilder, ContextBuilderError
from app.llm_client import LLMClient, LLMClientError
from app.prompt_builder import PromptBuilder, PromptBuilderError
from app.query import UserQuery
from app.response_processor import (
    GeneratedAnswer,
    ResponseProcessorError,
    process_response,
)
from app.retriever import RetrievalResult, Retriever, RetrieverError


class RAGPipelineError(Exception):
    """Raised when a RAG pipeline stage fails."""


@dataclass(frozen=True)
class PipelineTimings:
    """Elapsed time for each completed pipeline stage."""

    total_seconds: float
    retrieval_seconds: float
    context_seconds: float
    prompt_seconds: float
    generation_seconds: float
    response_processing_seconds: float


class RAGPipeline:
    """Run retrieval, context preparation, generation, and response processing."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_client: LLMClient | None = None,
        response_processor: Callable = process_response,
    ) -> None:
        self._retriever = retriever if retriever is not None else Retriever()
        self._context_builder = (
            context_builder if context_builder is not None else ContextBuilder()
        )
        self._prompt_builder = (
            prompt_builder if prompt_builder is not None else PromptBuilder()
        )
        self._llm_client = llm_client if llm_client is not None else LLMClient()
        self._response_processor = response_processor
        self.last_timings: PipelineTimings | None = None

    def run(self, query: UserQuery) -> GeneratedAnswer:
        """Run the full RAG workflow for a validated query."""

        self.last_timings = None
        total_started = perf_counter()
        elapsed: dict[str, float] = {}

        retrieval = self._run_stage(
            "retrieval",
            lambda: self._retriever.retrieve(query),
            elapsed,
            "retrieval_seconds",
        )
        if not isinstance(retrieval, RetrievalResult):
            raise RAGPipelineError("Pipeline retrieval returned an invalid result")
        if not retrieval.chunks:
            elapsed["total_seconds"] = perf_counter() - total_started
            self.last_timings = PipelineTimings(
                total_seconds=elapsed["total_seconds"],
                retrieval_seconds=elapsed["retrieval_seconds"],
                context_seconds=0.0,
                prompt_seconds=0.0,
                generation_seconds=0.0,
                response_processing_seconds=0.0,
            )
            raise RAGPipelineError("Retrieval returned no chunks; generation skipped")

        context = self._run_stage(
            "context building",
            lambda: self._context_builder.build(retrieval),
            elapsed,
            "context_seconds",
        )
        prompt = self._run_stage(
            "prompt building",
            lambda: self._prompt_builder.build(query, context),
            elapsed,
            "prompt_seconds",
        )
        response = self._run_stage(
            "LLM generation",
            lambda: self._llm_client.generate(prompt),
            elapsed,
            "generation_seconds",
        )
        answer = self._run_stage(
            "response processing",
            lambda: self._response_processor(response, retrieval),
            elapsed,
            "response_processing_seconds",
        )

        elapsed["total_seconds"] = perf_counter() - total_started
        elapsed["generation_seconds"] = response.generation_seconds
        self.last_timings = PipelineTimings(**elapsed)
        return answer

    @staticmethod
    def _run_stage(
        name: str,
        operation: Callable,
        timings: dict[str, float],
        timing_name: str,
    ):
        started = perf_counter()
        try:
            result = operation()
        except Exception as error:
            timings[timing_name] = perf_counter() - started
            raise RAGPipelineError(f"Pipeline {name} failed") from error
        timings[timing_name] = perf_counter() - started
        return result
