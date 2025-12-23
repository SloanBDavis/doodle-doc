from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from doodle_doc.core.models import SearchResult


@dataclass
class RetrievalMetrics:
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    num_queries: int = 0


@dataclass
class LatencyMetrics:
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0
    num_samples: int = 0


@dataclass
class EvalMetrics:
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    search_mode: str = "fast"
    timestamp: str = ""


def compute_recall_at_k(
    results: list[SearchResult],
    ground_truth_doc_id: str,
    ground_truth_page_num: int,
    k: int,
) -> float:
    """Check if ground truth appears in top-k results. Returns 1.0 if hit, 0.0 otherwise."""
    for result in results[:k]:
        if result.doc_id == ground_truth_doc_id and result.page_num == ground_truth_page_num:
            return 1.0
    return 0.0


def compute_mrr(
    results: list[SearchResult],
    ground_truth_doc_id: str,
    ground_truth_page_num: int,
) -> float:
    """Compute reciprocal rank for a single query. Returns 0.0 if not found."""
    for i, result in enumerate(results):
        if result.doc_id == ground_truth_doc_id and result.page_num == ground_truth_page_num:
            return 1.0 / (i + 1)
    return 0.0


def aggregate_retrieval_metrics(
    recalls: dict[int, list[float]],
    mrrs: list[float],
) -> RetrievalMetrics:
    """Aggregate per-query metrics into overall metrics."""
    num_queries = len(mrrs)
    if num_queries == 0:
        return RetrievalMetrics()

    return RetrievalMetrics(
        recall_at_1=float(np.mean(recalls.get(1, [0.0]))),
        recall_at_5=float(np.mean(recalls.get(5, [0.0]))),
        recall_at_10=float(np.mean(recalls.get(10, [0.0]))),
        recall_at_20=float(np.mean(recalls.get(20, [0.0]))),
        mrr=float(np.mean(mrrs)),
        num_queries=num_queries,
    )


def compute_latency_metrics(latencies_ms: list[float]) -> LatencyMetrics:
    """Compute latency percentiles from a list of latency measurements."""
    if not latencies_ms:
        return LatencyMetrics()

    arr = np.array(latencies_ms)
    return LatencyMetrics(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        mean_ms=float(np.mean(arr)),
        num_samples=len(latencies_ms),
    )


class LatencyTimer:
    """Context manager for timing operations."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
