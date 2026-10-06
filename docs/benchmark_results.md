### Embedding drift noise = 0.0

| Metric | Baseline RAG | RAG-Guard | Change |
|---|---|---|---|
| Recall | 100.0% | 100.0% | +0.0% |
| Answer accuracy | 100.0% | 100.0% | +0.0% |
| Faithfulness | 100.0% | 100.0% | +0.0% |
| Hallucination rate | 0.0% | 0.0% | n/a |
| Latency p50 (ms) | 4.09 | 4.02 | -1.8% |
| Latency p95 (ms) | 5.13 | 5.21 | +1.6% |
| Cost / query (USD) | 0.000522 | 0.000522 | +0.0% |
| Recovery success rate | n/a (none degraded) | n/a (none degraded) | n/a |

### Embedding drift noise = 3.0

| Metric | Baseline RAG | RAG-Guard | Change |
|---|---|---|---|
| Recall | 91.7% | 100.0% | +9.1% |
| Answer accuracy | 91.7% | 100.0% | +9.1% |
| Faithfulness | 100.0% | 100.0% | +0.0% |
| Hallucination rate | 0.0% | 0.0% | n/a |
| Latency p50 (ms) | 4.01 | 4.30 | +7.3% |
| Latency p95 (ms) | 5.29 | 5.54 | +4.8% |
| Cost / query (USD) | 0.000522 | 0.000522 | +0.0% |
| Recovery success rate | 0.0% (5 degraded) | 100.0% (5 degraded) | n/a |

### Embedding drift noise = 5.0

| Metric | Baseline RAG | RAG-Guard | Change |
|---|---|---|---|
| Recall | 71.7% | 100.0% | +39.5% |
| Answer accuracy | 71.7% | 100.0% | +39.5% |
| Faithfulness | 98.3% | 100.0% | +1.7% |
| Hallucination rate | 1.7% | 0.0% | -100.0% |
| Latency p50 (ms) | 4.21 | 4.10 | -2.6% |
| Latency p95 (ms) | 5.38 | 5.44 | +1.2% |
| Cost / query (USD) | 0.000519 | 0.000518 | -0.2% |
| Recovery success rate | 0.0% (17 degraded) | 100.0% (17 degraded) | n/a |

### Embedding drift noise = 8.0

| Metric | Baseline RAG | RAG-Guard | Change |
|---|---|---|---|
| Recall | 43.3% | 100.0% | +130.8% |
| Answer accuracy | 43.3% | 100.0% | +130.8% |
| Faithfulness | 83.3% | 100.0% | +20.0% |
| Hallucination rate | 16.7% | 0.0% | -100.0% |
| Latency p50 (ms) | 4.51 | 5.02 | +11.3% |
| Latency p95 (ms) | 5.60 | 6.75 | +20.5% |
| Cost / query (USD) | 0.000516 | 0.000517 | +0.1% |
| Recovery success rate | 0.0% (34 degraded) | 100.0% (34 degraded) | n/a |

### Embedding drift noise = 12.0

| Metric | Baseline RAG | RAG-Guard | Change |
|---|---|---|---|
| Recall | 25.0% | 100.0% | +300.0% |
| Answer accuracy | 25.0% | 100.0% | +300.0% |
| Faithfulness | 73.3% | 100.0% | +36.4% |
| Hallucination rate | 26.7% | 0.0% | -100.0% |
| Latency p50 (ms) | 3.95 | 3.21 | -18.7% |
| Latency p95 (ms) | 5.49 | 5.11 | -6.9% |
| Cost / query (USD) | 0.000514 | 0.000519 | +0.9% |
| Recovery success rate | 0.0% (45 degraded) | 100.0% (45 degraded) | n/a |
