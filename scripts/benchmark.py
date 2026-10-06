"""Run baseline vs RAG-Guard on the synthetic dataset and render a comparison chart.

The primary retriever is a dense index whose query embeddings are perturbed by deterministic
noise (simulating embedding drift); ``--noise`` sweeps the drift magnitude. The LLM is ``ExtractiveMockLLM``: results measure the pipeline's behaviour,
not a real model's. Usage: python scripts/benchmark.py [--noise 0 3 5 8 12] [--entities 60] [--out DIR]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from mocks.mock_llms import ExtractiveMockLLM
from mocks.mock_retrievers import DriftedDenseRetriever
from rag_guard.config import Settings
from rag_guard.evaluation.benchmark import BenchmarkRunner, comparison_table
from rag_guard.evaluation.dataset import QADataset
from rag_guard.orchestration.builder import build_pipeline
from rag_guard.types import BenchmarkResult


async def run_benchmark(noise: float, entities: int) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Return (baseline, guarded) results on identical data, retriever and LLM."""
    dataset = QADataset.synthetic(n_entities=entities)
    settings = Settings(_env_file=None)
    runner = BenchmarkRunner()
    results = []
    for name, guarded in (("Baseline RAG", False), ("RAG-Guard", True)):
        pipeline = build_pipeline(
            dataset.corpus,
            ExtractiveMockLLM(),
            settings,
            primary=DriftedDenseRetriever(dataset.corpus, noise=noise),
            guarded=guarded,
        )
        results.append(await runner.run(name, pipeline, dataset))
    return results[0], results[1]


def render_chart(baseline: BenchmarkResult, guarded: BenchmarkResult, path: Path) -> None:
    """Save a grouped bar chart of the quality metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Recall", "Answer accuracy", "Faithfulness", "Hallucination rate"]
    b, g = baseline.metrics, guarded.metrics
    base_vals = [b.recall, b.answer_accuracy, b.faithfulness, b.hallucination_rate]
    guard_vals = [g.recall, g.answer_accuracy, g.faithfulness, g.hallucination_rate]
    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_a = ax.bar(
        [i - width / 2 for i in x], base_vals, width, label=baseline.approach_name, color="#8a8f98"
    )
    bars_b = ax.bar(
        [i + width / 2 for i in x], guard_vals, width, label=guarded.approach_name, color="#2a6fdb"
    )
    for bar in [*bars_a, *bars_b]:
        ax.annotate(
            f"{bar.get_height():.0%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Baseline RAG vs RAG-Guard (synthetic benchmark)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 3.0, 5.0, 8.0, 12.0])
    parser.add_argument("--headline", type=float, default=5.0, help="noise level charted")
    parser.add_argument("--entities", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sections = []
    for noise in args.noise:
        baseline, guarded = asyncio.run(run_benchmark(noise, args.entities))
        table = comparison_table(baseline, guarded)
        sections.append(f"### Embedding drift noise = {noise}\n\n{table}\n")
        if noise == args.headline:
            render_chart(baseline, guarded, args.out / "benchmark_results.png")
    report = "\n".join(sections)
    print(report)
    (args.out / "benchmark_results.md").write_text(report, encoding="utf-8")
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
