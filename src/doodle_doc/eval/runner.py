from __future__ import annotations

import json
import logging
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
from doodle_doc.eval.pseudo_queries import PseudoQueryConfig, PseudoQueryGenerator
from doodle_doc.search.retrieval import SearchService

logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(
        self,
        settings: Settings,
        num_queries: int = 100,
        seed: int = 42,
        regenerate: bool = False,
    ) -> None:
        self.settings = settings
        self.num_queries = num_queries
        self.seed = seed
        self.regenerate = regenerate
        self._search_service: SearchService | None = None

    @property
    def search_service(self) -> SearchService:
        if self._search_service is None:
            self._search_service = SearchService(self.settings)
        return self._search_service

    @property
    def eval_dir(self) -> Path:
        return self.settings.data_dir / "eval"

    @property
    def pseudo_queries_dir(self) -> Path:
        return self.eval_dir / "pseudo_queries"

    @property
    def results_dir(self) -> Path:
        return self.eval_dir / "results"

    def run(self, modes: list[str] | None = None) -> dict[str, EvalMetrics]:
        """Run evaluation for specified search modes."""
        modes = modes or ["fast", "accurate"]
        self._ensure_pseudo_queries()

        results: dict[str, EvalMetrics] = {}
        for mode in modes:
            logger.info(f"Running evaluation for {mode} mode...")
            metrics = self._run_single_mode(mode)
            self._save_results(metrics, mode)
            results[mode] = metrics

        return results

    def _ensure_pseudo_queries(self) -> None:
        """Generate pseudo-queries if they don't exist or regenerate is requested."""
        gt_path = self.pseudo_queries_dir / "ground_truth.json"
        if gt_path.exists() and not self.regenerate:
            return

        logger.info("Generating pseudo-queries...")
        config = PseudoQueryConfig(
            num_queries=self.num_queries,
            seed=self.seed,
        )
        generator = PseudoQueryGenerator(self.settings, config)
        generator.generate(self.pseudo_queries_dir)

    def _run_single_mode(self, search_mode: str) -> EvalMetrics:
        """Run evaluation for a single search mode."""
        ground_truth = PseudoQueryGenerator.load_ground_truth(self.eval_dir)
        queries_dir = self.pseudo_queries_dir / "queries"

        self._warmup(search_mode)

        recalls: dict[int, list[float]] = {1: [], 5: [], 10: [], 20: []}
        mrrs: list[float] = []
        latencies: list[float] = []

        for query_id, gt in ground_truth.items():
            query_path = queries_dir / f"{query_id}.png"
            if not query_path.exists():
                continue

            query_img = Image.open(query_path)

            with LatencyTimer() as timer:
                results = self.search_service.search(
                    sketch_image=query_img,
                    top_k=20,
                    search_mode=search_mode,
                )

            latencies.append(timer.elapsed_ms)

            gt_doc_id = str(gt["doc_id"])
            gt_page_num = int(gt["page_num"])

            for k in [1, 5, 10, 20]:
                recalls[k].append(
                    compute_recall_at_k(results, gt_doc_id, gt_page_num, k)
                )
            mrrs.append(compute_mrr(results, gt_doc_id, gt_page_num))

        return EvalMetrics(
            retrieval=aggregate_retrieval_metrics(recalls, mrrs),
            latency=compute_latency_metrics(latencies),
            search_mode=search_mode,
            timestamp=datetime.now().isoformat(),
        )

    def _warmup(self, search_mode: str) -> None:
        """Run warmup queries to ensure models are loaded."""
        queries_dir = self.pseudo_queries_dir / "queries"
        query_files = list(queries_dir.glob("*.png"))[:3]

        for query_file in query_files:
            query_img = Image.open(query_file)
            self.search_service.search(
                sketch_image=query_img,
                top_k=5,
                search_mode=search_mode,
            )

    def _save_results(self, metrics: EvalMetrics, search_mode: str) -> Path:
        """Save evaluation results to disk."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.results_dir / f"{timestamp}_{search_mode}.json"

        with open(result_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        return result_path

    def save_as_baseline(self, search_mode: str) -> None:
        """Save current results as baseline for CI regression."""
        latest = self._get_latest_result(search_mode)
        if latest is None:
            raise ValueError(f"No results found for {search_mode} mode")

        baseline_path = self.results_dir / f"baseline_{search_mode}.json"
        with open(latest) as src, open(baseline_path, "w") as dst:
            dst.write(src.read())

        logger.info(f"Saved baseline for {search_mode} mode to {baseline_path}")

    def _get_latest_result(self, search_mode: str) -> Path | None:
        """Get the most recent result file for a search mode."""
        if not self.results_dir.exists():
            return None

        results = sorted(
            self.results_dir.glob(f"*_{search_mode}.json"),
            reverse=True,
        )
        for result in results:
            if not result.name.startswith("baseline_"):
                return result
        return None

    @classmethod
    def load_baseline(cls, results_dir: Path, search_mode: str) -> EvalMetrics | None:
        """Load baseline metrics for comparison."""
        baseline_path = results_dir / f"baseline_{search_mode}.json"
        if not baseline_path.exists():
            return None

        with open(baseline_path) as f:
            data = json.load(f)

        return EvalMetrics(
            retrieval=RetrievalMetrics(**data["retrieval"]),
            latency=LatencyMetrics(**data["latency"]),
            search_mode=data["search_mode"],
            timestamp=data["timestamp"],
        )

    def compare_to_baseline(
        self,
        current: EvalMetrics,
        search_mode: str,
        threshold: float = 0.05,
    ) -> tuple[bool, str]:
        """Compare current metrics to baseline. Returns (passed, message)."""
        baseline = self.load_baseline(self.results_dir, search_mode)
        if baseline is None:
            return True, f"No baseline found for {search_mode} mode, skipping comparison"

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
