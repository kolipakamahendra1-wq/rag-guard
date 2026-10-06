"""Evaluation and benchmarking."""

from rag_guard.evaluation.benchmark import BenchmarkRunner, comparison_table
from rag_guard.evaluation.dataset import QADataset
from rag_guard.evaluation.metrics import aggregate, answer_correct, recall_at_k

__all__ = [
    "BenchmarkRunner",
    "QADataset",
    "aggregate",
    "answer_correct",
    "comparison_table",
    "recall_at_k",
]
