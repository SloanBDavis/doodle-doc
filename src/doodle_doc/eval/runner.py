from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.eval.metrics import (
    EvalMetrics,
    LatencyMetrics,
    LatencyTimer,
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    compute_latency_metrics,
    compute_mrr,
    compute_recall_at_k,
)
from doodle_doc.ingestion.colqwen_embed import ColQwen2Embedder
from doodle_doc.ingestion.colqwen_index import ColQwen2Index
from doodle_doc.ingestion.colqwen_utils import maxsim_score


class EvalRunner:
    def __init__(
        self,
        settings: Settings,
        synth_dir: Path,
        top_k: int = 20,
        embedder: ColQwen2Embedder | None = None,
        index: ColQwen2Index | None = None,
    ) -> None:
        self.settings = settings
        self.synth_dir = synth_dir
        self.top_k = top_k
        self._embedder = embedder
        self._index = index

    @property
    def embedder(self) -> ColQwen2Embedder:
        if self._embedder is None:
            self._embedder = ColQwen2Embedder(model_name=self.settings.colqwen_model)
        return self._embedder

    @property
    def index(self) -> ColQwen2Index:
        if self._index is None:
            self._index = ColQwen2Index.load(self.synth_dir / "index" / "colqwen")
        return self._index

    @property
    def results_dir(self) -> Path:
        return self.synth_dir / "eval" / "results"

    def run(self) -> EvalMetrics:
        ground_truth = self._load_ground_truth()
        doodles_dir = self.synth_dir / "doodles"

        recalls: dict[int, list[float]] = {1: [], 5: [], 10: [], 20: []}
        mrrs: list[float] = []
        latencies: list[float] = []

        for doodle_id, gt in ground_truth.items():
            img_path = doodles_dir / f"{doodle_id}.png"
            if not img_path.exists():
                continue

            query_img = Image.open(img_path)

            with LatencyTimer() as timer:
                results = self.search(query_img)

            latencies.append(timer.elapsed_ms)

            target_doc_id = str(gt["page_id"])
            target_page_num = 0

            for k in [1, 5, 10, 20]:
                recalls[k].append(compute_recall_at_k(results, target_doc_id, target_page_num, k))
            mrrs.append(compute_mrr(results, target_doc_id, target_page_num))

        metrics = EvalMetrics(
            retrieval=aggregate_retrieval_metrics(recalls, mrrs),
            latency=compute_latency_metrics(latencies),
            timestamp=datetime.now().isoformat(),
        )
        self._save_results(metrics)
        return metrics

    def search(self, sketch_image: Image.Image) -> list[_EvalResult]:
        if not self.embedder.is_loaded():
            self.embedder.load()

        query_emb = self.embedder.embed_single(sketch_image.convert("RGB"))

        scores: list[tuple[str, int, float]] = []
        for doc_id, page_num in self.index.all_page_keys():
            doc_emb = self.index.get(doc_id, page_num)
            if doc_emb is None:
                continue
            score = maxsim_score(query_emb, doc_emb)
            if score == float("-inf"):
                continue
            scores.append((doc_id, page_num, score))

        scores.sort(key=lambda item: item[2], reverse=True)
        return [
            _EvalResult(doc_id=doc_id, page_num=page_num, score=float(score))
            for doc_id, page_num, score in scores[: self.top_k]
        ]

    def _save_results(self, metrics: EvalMetrics) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.results_dir / f"{timestamp}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)
        return result_path

    def save_as_baseline(self) -> None:
        latest = self._get_latest_result()
        if latest is None:
            raise ValueError("No results found")

        baseline_path = self.results_dir / "baseline.json"
        with open(latest) as src, open(baseline_path, "w") as dst:
            dst.write(src.read())

    def _get_latest_result(self) -> Path | None:
        if not self.results_dir.exists():
            return None

        results = sorted(self.results_dir.glob("*.json"), reverse=True)
        for result in results:
            if result.name != "baseline.json":
                return result
        return None

    @classmethod
    def load_baseline(cls, results_dir: Path) -> EvalMetrics | None:
        baseline_path = results_dir / "baseline.json"
        if not baseline_path.exists():
            return None

        with open(baseline_path) as f:
            data = json.load(f)

        return EvalMetrics(
            retrieval=RetrievalMetrics(**data["retrieval"]),
            latency=LatencyMetrics(**data["latency"]),
            timestamp=data["timestamp"],
        )

    def compare_to_baseline(
        self,
        current: EvalMetrics,
        threshold: float = 0.05,
    ) -> tuple[bool, str]:
        baseline = self.load_baseline(self.results_dir)
        if baseline is None:
            return True, "No baseline found, skipping comparison"

        baseline_recall = baseline.retrieval.recall_at_10
        current_recall = current.retrieval.recall_at_10
        diff = baseline_recall - current_recall

        if diff > threshold:
            return False, (
                f"REGRESSION: Recall@10 dropped from {baseline_recall:.3f} "
                f"to {current_recall:.3f} (diff: {diff:.3f}, threshold: {threshold})"
            )

        return True, (
            f"OK: Recall@10 = {current_recall:.3f} "
            f"(baseline: {baseline_recall:.3f}, diff: {-diff:.3f})"
        )

    def _load_ground_truth(self) -> dict[str, dict[str, str]]:
        gt_path = self.synth_dir / "ground_truth.json"
        with open(gt_path) as f:
            data: dict[str, dict[str, str]] = json.load(f)
            return data


class _EvalResult:
    def __init__(self, doc_id: str, page_num: int, score: float) -> None:
        self.doc_id = doc_id
        self.page_num = page_num
        self.score = score
        self.doc_name = f"{doc_id}.png"
        self.thumbnail_url = ""
