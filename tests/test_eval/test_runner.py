from __future__ import annotations

from pathlib import Path

import pytest

from doodle_doc.core.config import Settings
from doodle_doc.eval.metrics import EvalMetrics, LatencyMetrics, RetrievalMetrics
from doodle_doc.eval.runner import EvalRunner


class TestCompareToBaseline:
    @pytest.fixture
    def settings(self, temp_dir: Path) -> Settings:
        return Settings(data_dir=temp_dir)

    def test_passes_when_no_regression(self, settings: Settings) -> None:
        runner = EvalRunner(settings)

        current = EvalMetrics(
            retrieval=RetrievalMetrics(recall_at_10=0.75),
            latency=LatencyMetrics(),
            search_mode="fast",
            timestamp="2024-01-01",
        )

        runner.results_dir.mkdir(parents=True, exist_ok=True)

        import json
        baseline = EvalMetrics(
            retrieval=RetrievalMetrics(recall_at_10=0.72),
            latency=LatencyMetrics(),
            search_mode="fast",
            timestamp="2024-01-01",
        )
        from dataclasses import asdict
        with open(runner.results_dir / "baseline_fast.json", "w") as f:
            json.dump(asdict(baseline), f)

        passed, message = runner.compare_to_baseline(current, "fast", threshold=0.05)
        assert passed is True
        assert "OK" in message

    def test_fails_when_regression(self, settings: Settings) -> None:
        runner = EvalRunner(settings)

        current = EvalMetrics(
            retrieval=RetrievalMetrics(recall_at_10=0.65),
            latency=LatencyMetrics(),
            search_mode="fast",
            timestamp="2024-01-01",
        )

        runner.results_dir.mkdir(parents=True, exist_ok=True)

        import json
        baseline = EvalMetrics(
            retrieval=RetrievalMetrics(recall_at_10=0.72),
            latency=LatencyMetrics(),
            search_mode="fast",
            timestamp="2024-01-01",
        )
        from dataclasses import asdict
        with open(runner.results_dir / "baseline_fast.json", "w") as f:
            json.dump(asdict(baseline), f)

        passed, message = runner.compare_to_baseline(current, "fast", threshold=0.05)
        assert passed is False
        assert "REGRESSION" in message

    def test_passes_when_no_baseline(self, settings: Settings) -> None:
        runner = EvalRunner(settings)

        current = EvalMetrics(
            retrieval=RetrievalMetrics(recall_at_10=0.65),
            latency=LatencyMetrics(),
            search_mode="fast",
            timestamp="2024-01-01",
        )

        passed, message = runner.compare_to_baseline(current, "fast")
        assert passed is True
        assert "No baseline" in message


class TestLoadBaseline:
    @pytest.fixture
    def settings(self, temp_dir: Path) -> Settings:
        return Settings(data_dir=temp_dir)

    def test_load_existing_baseline(self, settings: Settings) -> None:
        results_dir = settings.data_dir / "eval" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        import json
        from dataclasses import asdict
        baseline = EvalMetrics(
            retrieval=RetrievalMetrics(
                recall_at_1=0.5,
                recall_at_5=0.7,
                recall_at_10=0.8,
                recall_at_20=0.9,
                mrr=0.6,
                num_queries=100,
            ),
            latency=LatencyMetrics(p50_ms=100.0, p95_ms=200.0, mean_ms=150.0, num_samples=100),
            search_mode="fast",
            timestamp="2024-01-01T00:00:00",
        )
        with open(results_dir / "baseline_fast.json", "w") as f:
            json.dump(asdict(baseline), f)

        loaded = EvalRunner.load_baseline(results_dir, "fast")
        assert loaded is not None
        assert loaded.retrieval.recall_at_10 == 0.8
        assert loaded.latency.p50_ms == 100.0

    def test_load_nonexistent_baseline(self, settings: Settings) -> None:
        results_dir = settings.data_dir / "eval" / "results"
        loaded = EvalRunner.load_baseline(results_dir, "fast")
        assert loaded is None
