from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.core.database import Database


@dataclass
class PseudoQuery:
    query_id: str
    doc_id: str
    page_num: int
    crop_box: tuple[int, int, int, int]


@dataclass
class PseudoQueryConfig:
    num_queries: int = 100
    min_crop_ratio: float = 0.15
    max_crop_ratio: float = 0.40
    seed: int = 42
    exclude_margins_pct: float = 0.05


class PseudoQueryGenerator:
    def __init__(
        self,
        settings: Settings,
        config: PseudoQueryConfig | None = None,
    ) -> None:
        self.settings = settings
        self.config = config or PseudoQueryConfig()
        self._db: Database | None = None

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = Database(self.settings.index_dir / "metadata.sqlite")
        return self._db

    def generate(self, output_dir: Path) -> list[PseudoQuery]:
        """Generate pseudo-queries from indexed pages."""
        output_dir.mkdir(parents=True, exist_ok=True)
        queries_dir = output_dir / "queries"
        queries_dir.mkdir(exist_ok=True)

        rng = random.Random(self.config.seed)
        pages = self._sample_pages(rng)

        if len(pages) < self.config.num_queries:
            raise ValueError(
                f"Not enough pages in index ({len(pages)}) "
                f"for {self.config.num_queries} queries"
            )

        queries: list[PseudoQuery] = []
        for i, (doc_id, page_num) in enumerate(pages[: self.config.num_queries]):
            query_id = f"q{i:04d}"

            page_path = self.settings.rendered_dir / doc_id / f"{page_num}.png"
            if not page_path.exists():
                continue

            img = Image.open(page_path)
            crop, crop_box = self._extract_random_crop(img, rng)

            crop_path = queries_dir / f"{query_id}.png"
            crop.save(crop_path, "PNG")

            queries.append(PseudoQuery(
                query_id=query_id,
                doc_id=doc_id,
                page_num=page_num,
                crop_box=crop_box,
            ))

        self._save_manifest(output_dir, queries)
        self._save_ground_truth(output_dir, queries)

        return queries

    def _sample_pages(self, rng: random.Random) -> list[tuple[str, int]]:
        """Sample random pages from the index."""
        docs = self.db.get_all_documents()
        all_pages: list[tuple[str, int]] = []

        for doc in docs:
            pages = self.db.get_pages_for_document(doc.doc_id)
            for page in pages:
                all_pages.append((doc.doc_id, page.page_num))

        rng.shuffle(all_pages)
        return all_pages

    def _extract_random_crop(
        self,
        img: Image.Image,
        rng: random.Random,
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        """Extract a random rectangular crop from the image."""
        w, h = img.size

        margin_x = int(w * self.config.exclude_margins_pct)
        margin_y = int(h * self.config.exclude_margins_pct)

        crop_ratio = rng.uniform(
            self.config.min_crop_ratio,
            self.config.max_crop_ratio,
        )

        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)

        max_x = w - margin_x - crop_w
        max_y = h - margin_y - crop_h

        x0 = rng.randint(margin_x, max(margin_x, max_x))
        y0 = rng.randint(margin_y, max(margin_y, max_y))
        x1 = x0 + crop_w
        y1 = y0 + crop_h

        crop = img.crop((x0, y0, x1, y1))
        return crop, (x0, y0, x1, y1)

    def _save_manifest(
        self,
        output_dir: Path,
        queries: list[PseudoQuery],
    ) -> None:
        """Save manifest with generation parameters."""
        manifest = {
            "version": 1,
            "generated_at": datetime.now().isoformat(),
            "config": asdict(self.config),
            "num_queries": len(queries),
        }
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _save_ground_truth(
        self,
        output_dir: Path,
        queries: list[PseudoQuery],
    ) -> None:
        """Save ground truth mapping."""
        ground_truth = {
            q.query_id: {
                "doc_id": q.doc_id,
                "page_num": q.page_num,
                "crop_box": q.crop_box,
            }
            for q in queries
        }
        with open(output_dir / "ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=2)

    @classmethod
    def load_ground_truth(cls, eval_dir: Path) -> dict[str, dict[str, Any]]:
        """Load ground truth from disk."""
        gt_path = eval_dir / "pseudo_queries" / "ground_truth.json"
        with open(gt_path) as f:
            data: dict[str, dict[str, Any]] = json.load(f)
            return data
