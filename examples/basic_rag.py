"""Compare a baseline pipeline with RAG-Guard when the primary retriever has drifted.

Run: python examples/basic_rag.py
"""

from __future__ import annotations

import asyncio

from mocks.mock_llms import ExtractiveMockLLM
from mocks.mock_retrievers import DriftedDenseRetriever
from rag_guard.config import Settings
from rag_guard.evaluation.dataset import QADataset
from rag_guard.orchestration.builder import build_pipeline


async def main() -> None:
    dataset = QADataset.synthetic()
    settings = Settings(_env_file=None)
    question = dataset.pairs[3].question
    print(f"Question: {question}\nExpected: {dataset.pairs[3].expected_answer}\n")
    for label, guarded in (("baseline ", False), ("rag-guard", True)):
        pipeline = build_pipeline(
            dataset.corpus,
            ExtractiveMockLLM(),
            settings,
            primary=DriftedDenseRetriever(dataset.corpus, noise=8.0),
            guarded=guarded,
        )
        result = await pipeline.run(question)
        print(
            f"[{label}] answer={result.answer!r}\n"
            f"            fallback={result.fallback_used.value} attempts={result.fallback_attempts} "
            f"faithfulness={result.faithfulness}"
        )


if __name__ == "__main__":
    asyncio.run(main())
