from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.eval.metrics import EvalMetrics, LatencyMetrics, RetrievalMetrics
from doodle_doc.eval.runner import EvalRunner


@pytest.fixture
def synth_dir(temp_dir: Path) -> Path:
    path = temp_dir / "synth"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def settings(temp_dir: Path) -> Settings:
    return Settings(data_dir=temp_dir / "data")


class TestCompareToBaseline:
    def test_passes_when_no_regression(self, settings: Settings, synth_dir: Path) -> None:
        runner = EvalRunner(settings, synth_dir=synth_dir)

        current = EvalMetrics(
            retrieval=RetrievalMetrics(recall_at_10=0.75),
            latency=LatencyMetrics(),
            timestamp="2024-01-01",
        )

        runner.results_dir.mkdir(parents=True, exist_ok=True)
        baseline = EvalMetrics(
            retrieval=RetrievalMetrics(recall_at_10=0.72),
            latency=LatencyMetrics(),
            timestamp="2024-01-01",
        )
        with open(runner.results_dir / "baseline.json", "w") as f:
            json.dump(asdict(baseline), f)

        passed, message = runner.compare_to_baseline(current, threshold=0.05)
        assert passed is True
        assert "OK" in message

    def test_fails_when_regression(self, settings: Settings, synth_dir: Path) -> None:
        runner = EvalRunner(settings, synth_dir=synth_dir)

        current = EvalMetrics(
            retrieval=RetrievalMetrics(recall_at_10=0.65),
            latency=LatencyMetrics(),
            timestamp="2024-01-01",
        )

        runner.results_dir.mkdir(parents=True, exist_ok=True)
        baseline = EvalMetrics(
            retrieval=RetrievalMetrics(recall_at_10=0.72),
            latency=LatencyMetrics(),
            timestamp="2024-01-01",
        )
        with open(runner.results_dir / "baseline.json", "w") as f:
            json.dump(asdict(baseline), f)

        passed, message = runner.compare_to_baseline(current, threshold=0.05)
        assert passed is False
        assert "REGRESSION" in message

    def test_passes_when_no_baseline(self, settings: Settings, synth_dir: Path) -> None:
        runner = EvalRunner(settings, synth_dir=synth_dir)

        current = EvalMetrics(
            retrieval=RetrievalMetrics(recall_at_10=0.65),
            latency=LatencyMetrics(),
            timestamp="2024-01-01",
        )

        passed, message = runner.compare_to_baseline(current)
        assert passed is True
        assert "No baseline" in message


class TestLoadBaseline:
    def test_load_existing_baseline(self, synth_dir: Path) -> None:
        results_dir = synth_dir / "eval" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

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
            timestamp="2024-01-01T00:00:00",
        )
        with open(results_dir / "baseline.json", "w") as f:
            json.dump(asdict(baseline), f)

        loaded = EvalRunner.load_baseline(results_dir)
        assert loaded is not None
        assert loaded.retrieval.recall_at_10 == 0.8
        assert loaded.latency.p50_ms == 100.0

    def test_load_nonexistent_baseline(self, synth_dir: Path) -> None:
        results_dir = synth_dir / "eval" / "results"
        loaded = EvalRunner.load_baseline(results_dir)
        assert loaded is None


class _FakeEmbedder:
    def __init__(self, query_emb: np.ndarray) -> None:
        self.query_emb = query_emb

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def embed_single(self, img: Image.Image) -> np.ndarray:
        return self.query_emb


class _FakeIndex:
    def __init__(self, embeddings: dict[tuple[str, int], np.ndarray]) -> None:
        self.embeddings = embeddings

    def all_page_keys(self) -> list[tuple[str, int]]:
        return list(self.embeddings)

    def get(self, doc_id: str, page_num: int) -> np.ndarray | None:
        return self.embeddings.get((doc_id, page_num))


def test_search_handles_zero_padded_doc_embeddings(
    settings: Settings,
    synth_dir: Path,
) -> None:
    runner = EvalRunner(
        settings,
        synth_dir=synth_dir,
        embedder=_FakeEmbedder(np.array([[1.0, 0.0]], dtype=np.float32)),
        index=_FakeIndex(
            {
                ("page_bad", 0): np.array(
                    [[0.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
                    dtype=np.float32,
                ),
                ("page_good", 0): np.array([[1.0, 0.0]], dtype=np.float32),
            }
        ),
    )

    results = runner.search(Image.new("RGB", (8, 8), "white"))

    assert [result.doc_id for result in results[:2]] == ["page_good", "page_bad"]
    assert np.isfinite(results[0].score)
