from __future__ import annotations

import time

import pytest

from doodle_doc.core.models import SearchResult
from doodle_doc.eval.metrics import (
    LatencyTimer,
    compute_latency_metrics,
    compute_mrr,
    compute_recall_at_k,
    aggregate_retrieval_metrics,
)


def make_result(doc_id: str, page_num: int, score: float) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        doc_name=f"{doc_id}.pdf",
        page_num=page_num,
        score=score,
        stage="fast",
        thumbnail_url=f"/v1/thumb/{doc_id}/{page_num}",
    )


class TestRecallAtK:
    def test_hit_at_position_1(self) -> None:
        results = [
            make_result("doc1", 5, 0.9),
            make_result("doc2", 3, 0.8),
            make_result("doc3", 1, 0.7),
        ]
        assert compute_recall_at_k(results, "doc1", 5, k=1) == 1.0
        assert compute_recall_at_k(results, "doc1", 5, k=3) == 1.0

    def test_hit_at_position_3(self) -> None:
        results = [
            make_result("doc1", 5, 0.9),
            make_result("doc2", 3, 0.8),
            make_result("doc3", 1, 0.7),
        ]
        assert compute_recall_at_k(results, "doc3", 1, k=1) == 0.0
        assert compute_recall_at_k(results, "doc3", 1, k=3) == 1.0

    def test_no_hit(self) -> None:
        results = [
            make_result("doc1", 5, 0.9),
            make_result("doc2", 3, 0.8),
        ]
        assert compute_recall_at_k(results, "doc4", 1, k=2) == 0.0

    def test_empty_results(self) -> None:
        assert compute_recall_at_k([], "doc1", 1, k=10) == 0.0


class TestMRR:
    def test_hit_at_position_1(self) -> None:
        results = [
            make_result("doc1", 5, 0.9),
            make_result("doc2", 3, 0.8),
        ]
        assert compute_mrr(results, "doc1", 5) == 1.0

    def test_hit_at_position_2(self) -> None:
        results = [
            make_result("doc1", 5, 0.9),
            make_result("doc2", 3, 0.8),
        ]
        assert compute_mrr(results, "doc2", 3) == 0.5

    def test_hit_at_position_3(self) -> None:
        results = [
            make_result("doc1", 5, 0.9),
            make_result("doc2", 3, 0.8),
            make_result("doc3", 1, 0.7),
        ]
        assert compute_mrr(results, "doc3", 1) == pytest.approx(1 / 3)

    def test_no_hit(self) -> None:
        results = [make_result("doc1", 5, 0.9)]
        assert compute_mrr(results, "doc4", 1) == 0.0

    def test_empty_results(self) -> None:
        assert compute_mrr([], "doc1", 1) == 0.0


class TestAggregateRetrievalMetrics:
    def test_aggregate(self) -> None:
        recalls = {
            1: [1.0, 0.0, 1.0],
            5: [1.0, 1.0, 1.0],
            10: [1.0, 1.0, 1.0],
            20: [1.0, 1.0, 1.0],
        }
        mrrs = [1.0, 0.5, 1.0]

        metrics = aggregate_retrieval_metrics(recalls, mrrs)

        assert metrics.recall_at_1 == pytest.approx(2 / 3)
        assert metrics.recall_at_5 == 1.0
        assert metrics.mrr == pytest.approx(5 / 6)
        assert metrics.num_queries == 3

    def test_empty(self) -> None:
        metrics = aggregate_retrieval_metrics({}, [])
        assert metrics.recall_at_1 == 0.0
        assert metrics.num_queries == 0


class TestLatencyMetrics:
    def test_percentiles(self) -> None:
        latencies = [10.0, 20.0, 30.0, 40.0, 100.0]
        metrics = compute_latency_metrics(latencies)
        assert metrics.p50_ms == 30.0
        assert metrics.p95_ms >= 80.0
        assert metrics.mean_ms == 40.0
        assert metrics.num_samples == 5

    def test_empty(self) -> None:
        metrics = compute_latency_metrics([])
        assert metrics.p50_ms == 0.0
        assert metrics.num_samples == 0


class TestLatencyTimer:
    def test_timer_measures_time(self) -> None:
        timer = LatencyTimer()
        with timer:
            time.sleep(0.01)
        assert timer.elapsed_ms >= 10.0
        assert timer.elapsed_ms < 100.0
