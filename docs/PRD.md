# PRD: RAG-Guard

| | |
|---|---|
| Status | v0.1 implemented; roadmap items open |
| Owner | Mahendra Kolipaka |
| Last updated | 2026-10-06 |

## 1. Summary

RAG-Guard is a Python library and small HTTP service that sits between retrieval and generation in a RAG pipeline. It detects degraded retrieval at request time and automatically recovers through fallback strategies, returning an auditable result.

## 2. Problem

RAG failures are silent. When retrieval quality drops (embedding drift, stale index, unusual phrasing), the LLM still produces a confident answer from poor context. Existing evaluation tooling is offline and post-hoc; teams discover problems from user complaints. There is no standard, testable in-request mechanism that detects the problem and tries alternatives before generating.

## 3. Goals

- G1: Detect degraded retrieval per request without ground-truth labels.
- G2: Recover automatically with bounded cost and latency.
- G3: Work across LLM providers and retrieval backends through narrow interfaces.
- G4: Make every decision auditable (strategies tried, scores, cost, latency).
- G5: Be fully testable offline and reproducible.

## 4. Non-goals

- Building a vector database or embedding model.
- Replacing offline evaluation frameworks (Ragas, DeepEval); RAG-Guard complements them.
- Prompt engineering, agent orchestration, or multi-turn memory.
- Guaranteeing answer correctness (see Risks).

## 5. Users

| Persona | Need |
|---|---|
| ML/AI engineer running RAG in production | Stop bad answers caused by retrieval drift; understand why a query fell back |
| Platform engineer | Provider portability, cost visibility, predictable failure behaviour |
| Evaluator/researcher | Reproducible benchmark of recovery strategies |

## 6. Functional requirements

| ID | Requirement | Status |
|---|---|---|
| FR-1 | Score retrieved chunks with label-free signals (coverage, best overlap, mean retriever score) | Done |
| FR-2 | Classify results as healthy / degraded / critical using a threshold and a rolling-baseline drift check | Done |
| FR-3 | Execute an ordered fallback chain with per-attempt timeout and exponential-backoff retries; keep the best result; stop when healthy | Done |
| FR-4 | Provide BM25, dense, keyword, and RRF hybrid retrievers plus a reranker, behind a `Retriever` interface | Done |
| FR-5 | Provide Anthropic, OpenAI, Azure OpenAI, and local OpenAI-compatible providers behind an `LLMProvider` interface, with typed errors and transient-only retries | Done |
| FR-6 | Return a `RAGResult` with answer, chunks, detection result, fallback used/attempts, cost, latency, faithfulness, and a state audit trail | Done |
| FR-7 | Never raise for backend or provider failure; return `success=False` with an error | Done |
| FR-8 | Enforce an optional per-query cost budget | Done (flags overrun after the call) |
| FR-9 | Reorder fallback stages by observed success (`AdaptiveRecovery`) | Done (global, not per query type) |
| FR-10 | Expose `POST /query` and `GET /health`; run in offline demo mode without API keys | Done |
| FR-11 | Provide a deterministic benchmark harness and comparison table | Done |
| FR-12 | True token streaming (SSE) | Roadmap |
| FR-13 | Learned (cross-encoder / LLM-judge) detector and entailment-based faithfulness | Roadmap |
| FR-14 | Adapters for real vector stores (pgvector, Qdrant, etc.) | Roadmap |

## 7. Non-functional requirements

- Python 3.11+, fully typed (`mypy` strict clean), lint-clean (`ruff`).
- Tests run with no network; coverage at least 90% enforced in CI (currently ~98%).
- Library overhead per query in the low milliseconds with in-memory backends.
- Secrets only via environment variables; never logged.
- Structured JSON logs with context binding.

## 8. Success metrics

| Metric | Target | Measured (synthetic, v0.1) |
|---|---|---|
| Recall under heavy drift vs baseline | Meaningful improvement | 43.3% to 100% at noise 8.0 |
| Added cost when retrieval is healthy | ~0 | 0.0% at noise 0.0 |
| Fallbacks triggered when healthy | 0 | 0 |
| Test coverage | at least 90% | ~98% |

Real-world targets (hallucination reduction on production traffic) are **not yet measured**; they require evaluation on real corpora and models.

## 9. Risks and open questions

- **R1 Faithfulness is not correctness.** A faithful answer from the wrong but similar document passes. Mitigation: track recall/accuracy offline; roadmap learned detector.
- **R2 Heuristic detector false positives/negatives.** Wrong chunks sharing query vocabulary look healthy. Mitigation: configurable thresholds; pluggable scorer.
- **R3 Synthetic benchmark optimism.** Unique entity tokens make BM25 recovery trivial. Mitigation: documented clearly; add public QA datasets.
- **R4 Fallbacks add latency/cost on degraded queries.** Mitigation: `max_attempts`, per-attempt timeout, early stop.
- Open: how should thresholds be calibrated per corpus? Should drift baselines be shared across processes (Redis)?

## 10. Roadmap

1. Evaluate on public QA sets (e.g. Natural Questions subset) with a real LLM; publish results.
2. Pluggable learned detector and entailment faithfulness.
3. Real vector-store adapters; SSE streaming.
4. Per-query-type adaptive recovery; shared drift state.
5. OpenTelemetry export.
