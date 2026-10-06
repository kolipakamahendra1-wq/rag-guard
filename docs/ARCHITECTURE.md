# Architecture

## Request flow

1. `RAGPipeline.run(query)` opens a `Telemetry` (trace id, timed spans) and a `StateManager` (append-only snapshots).
2. `FallbackChain.run(query)` executes the plan `[primary, *stages]`:
   - each step calls `retriever.retrieve` under `with_timeout`, wrapped by `retry_async` (exponential backoff);
   - an optional `Reranker` post-processes chunks;
   - `FailureDetector.detect` scores the chunks and classifies them;
   - the best-scoring attempt is retained; the loop stops at the first healthy result;
   - a step that raises is recorded in the audit trail and skipped.
3. If no chunks survive, the pipeline returns `success=False` without calling the LLM.
4. Otherwise `LLMProvider.generate` produces the answer; `faithfulness_score` grades it against the context.
5. `RAGResult` is returned with answer, chunks, detection, fallback used/attempts, cost, latency, and the state trail.

## Components

| Package | Responsibility |
|---|---|
| `detection` | `RetrievalScorer` (coverage 0.5, best overlap 0.3, mean score 0.2 by default), `FailureDetector` (threshold, critical threshold, rolling baseline drift), faithfulness metrics |
| `retrieval` | `Retriever` ABC; BM25, hashed-dense, keyword, RRF hybrid; `OverlapReranker`; `FallbackChain` |
| `llm` | `LLMProvider` template (`_call` per provider); Anthropic, OpenAI, Azure, Local; `create_provider` |
| `orchestration` | `RAGPipeline`, `StateManager`, `AdaptiveRecovery`, `build_pipeline` |
| `evaluation` | synthetic `QADataset`, metric aggregation, `BenchmarkRunner`, markdown comparison |
| `api` / `server` | FastAPI factory with injected pipeline; ASGI entrypoint with offline demo mode |

## Detector semantics

- Score below `critical_threshold` (or no chunks): **critical**, recommend an alternative retriever.
- Score below `threshold`, or more than `drift_threshold` below the rolling mean of recent healthy scores: **degraded**. Low term coverage recommends a broader search, otherwise rerank.
- Only healthy scores update the baseline, so a degradation episode cannot normalise itself.
- Drift state is per-detector and in-process.

## Error model

All library errors derive from `RAGGuardException` with a stable `code`. Provider errors split into `TransientLLMError` (429, 408, 5xx, network; retried) and `LLMProviderError` (other 4xx, malformed body; not retried). Pipeline-level failures never propagate as exceptions to callers of `run`.

## Testing strategy

All tests run offline. Providers are exercised through `httpx.MockTransport`; retrievers and LLMs through deterministic doubles in `src/mocks`. Embedding drift is simulated by `DriftedDenseRetriever` (query vector plus deterministic, query-seeded noise), so benchmark results are exactly reproducible.
