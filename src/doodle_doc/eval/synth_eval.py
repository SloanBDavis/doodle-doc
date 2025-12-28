from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.ingestion.colqwen_embed import ColQwen2Embedder
from doodle_doc.ingestion.colqwen_index import ColQwen2Index


@dataclass
class SynthEvalResult:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    num_queries: int

    def summary(self) -> str:
        lines = [
            f"Queries:     {self.num_queries}",
            "",
            f"Recall@1:    {self.recall_at_1:.3f}",
            f"Recall@5:    {self.recall_at_5:.3f}",
            f"Recall@10:   {self.recall_at_10:.3f}",
            f"MRR:         {self.mrr:.3f}",
            "",
            f"Latency p50: {self.latency_p50_ms:.0f}ms",
            f"Latency p95: {self.latency_p95_ms:.0f}ms",
        ]
        return "\n".join(lines)


class SynthEvalRunner:
    def __init__(self, settings: Settings, synth_dir: Path) -> None:
        self.settings = settings
        self.synth_dir = synth_dir
        self.index_dir = synth_dir / "index" / "colqwen"
        self._embedder: ColQwen2Embedder | None = None
        self._index: ColQwen2Index | None = None

    @property
    def embedder(self) -> ColQwen2Embedder:
        if self._embedder is None:
            self._embedder = ColQwen2Embedder(model_name=self.settings.colqwen_model)
        return self._embedder

    @property
    def index(self) -> ColQwen2Index:
        if self._index is None:
            self._index = ColQwen2Index.load(self.index_dir)
        return self._index

    def run(self, top_k: int = 20) -> SynthEvalResult:
        ground_truth = self._load_ground_truth()

        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"Synth index not found at {self.index_dir}. "
                f"Run 'doodle-doc synth-index {self.synth_dir}' first."
            )

        all_recalls: dict[int, list[float]] = {1: [], 5: [], 10: []}
        all_rrs: list[float] = []
        latencies: list[float] = []

        doodles_dir = self.synth_dir / "doodles"
        all_pages = self.index.all_page_keys()

        if not all_pages:
            raise ValueError("No pages in synth index. Run synth-index first.")

        for doodle_id, gt in ground_truth.items():
            img_path = doodles_dir / f"{doodle_id}.png"
            if not img_path.exists():
                print(f"  Warning: {img_path} not found, skipping")
                continue

            img = Image.open(img_path)

            start = time.perf_counter()
            result_ids = self._search(img, top_k)
            latencies.append((time.perf_counter() - start) * 1000)

            target_page = gt["page_id"]

            for k in [1, 5, 10]:
                hit = 1.0 if target_page in result_ids[:k] else 0.0
                all_recalls[k].append(hit)

            rr = 0.0
            for i, rid in enumerate(result_ids):
                if rid == target_page:
                    rr = 1.0 / (i + 1)
                    break
            all_rrs.append(rr)

        return SynthEvalResult(
            recall_at_1=float(np.mean(all_recalls[1])) if all_recalls[1] else 0.0,
            recall_at_5=float(np.mean(all_recalls[5])) if all_recalls[5] else 0.0,
            recall_at_10=float(np.mean(all_recalls[10])) if all_recalls[10] else 0.0,
            mrr=float(np.mean(all_rrs)) if all_rrs else 0.0,
            latency_p50_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
            latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            num_queries=len(ground_truth),
        )

    def _search(self, sketch_image: Image.Image, top_k: int) -> list[str]:
        """Search synth index and return list of page IDs."""
        query_emb = self.embedder.embed_single(sketch_image)
        query_tensor = torch.from_numpy(query_emb).unsqueeze(0)

        all_pages = self.index.all_page_keys()
        scores: list[tuple[str, float]] = []

        for doc_id, page_num in all_pages:
            doc_emb = self.index.get(doc_id, page_num)
            if doc_emb is None:
                continue

            doc_tensor = torch.from_numpy(doc_emb).unsqueeze(0)
            score = self._compute_maxsim(query_tensor, doc_tensor)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scores[:top_k]]

    def _compute_maxsim(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor,
    ) -> float:
        """Compute MaxSim late-interaction score."""
        query_norm = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
        doc_norm = torch.nn.functional.normalize(doc_emb, p=2, dim=-1)

        sim_matrix = torch.matmul(query_norm[0], doc_norm[0].T)
        max_sims = sim_matrix.max(dim=1).values
        return float(max_sims.sum().item())

    def _load_ground_truth(self) -> dict[str, dict[str, Any]]:
        gt_path = self.synth_dir / "ground_truth.json"
        with open(gt_path) as f:
            data: dict[str, dict[str, Any]] = json.load(f)
            return data
