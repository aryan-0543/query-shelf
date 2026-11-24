# Query Shelf RAG Pipeline

## Flow

`RAGPipeline.run(UserQuery)` coordinates the existing application components:

```text
UserQuery
  -> Retriever
  -> ContextBuilder
  -> PromptBuilder
  -> LLMClient
  -> process_response
  -> GeneratedAnswer
```

The pipeline owns orchestration only. Retrieval, context budgeting, prompt
construction, provider communication, and citation validation remain in their
existing components.

## Empty retrieval

When the retriever returns no chunks, the pipeline raises `RAGPipelineError`
immediately. It does not call context building, prompt construction, the LLM,
or response processing.

## Errors and timing

Failures are wrapped as `RAGPipelineError` with the failing stage named and
the original exception chained as the cause. `last_timings` is cleared at the
start of every run. On completion, it contains immutable total and stage
timings; generation time is taken from `LLMResponse.generation_seconds`.
For an empty retrieval failure, retrieval and total timings are available and
remaining stage timings are zero.

## Dependency injection

Retriever, context builder, prompt builder, LLM client, and response processor
can be supplied to `RAGPipeline` for testing or application composition.
Defaults instantiate the existing project components. The pipeline adds no
framework, provider logic, caching, reranking, or conversational state.
