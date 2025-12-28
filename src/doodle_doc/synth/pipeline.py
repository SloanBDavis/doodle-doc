from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from doodle_doc.synth.gemini_generator import SUBJECTS, GeminiGenerator


@dataclass
class SynthConfig:
    output_dir: Path = Path("data/synth")
    num_pairs: int = 25
    seed: int = 42


@dataclass
class SynthStats:
    pages: int
    doodles: int
    output_dir: Path


class SynthPipeline:
    def __init__(self, config: SynthConfig | None = None) -> None:
        self.config = config or SynthConfig()
        self._gemini: GeminiGenerator | None = None
        self._rng = random.Random(self.config.seed)

    @property
    def gemini(self) -> GeminiGenerator:
        if self._gemini is None:
            self._gemini = GeminiGenerator()
        return self._gemini

    def run(self) -> SynthStats:
        self._setup_dirs()

        pages_dir = self.config.output_dir / "pages"
        doodles_dir = self.config.output_dir / "doodles"
        ground_truth = self._load_existing_ground_truth()
        start_idx = self._find_next_index(ground_truth)

        existing_pairs = 0
        if ground_truth:
            print(f"Found {len(ground_truth)} existing pairs, appending from index {start_idx}")
            existing_pairs = len(ground_truth)

        for i in range(self.config.num_pairs):
            idx = start_idx + i
            subject = self._rng.choice(SUBJECTS)
            page_id = f"page_{idx:04d}"
            doodle_id = f"doodle_{idx:04d}"

            print(f"[{i+1}/{self.config.num_pairs}] {subject[:40]}...")

            page = self.gemini.generate_notes_page(subject)
            page.save(pages_dir / f"{page_id}.png", "PNG")

            doodle, element = self.gemini.generate_doodle_for_page(page)
            doodle.save(doodles_dir / f"{doodle_id}.png", "PNG")

            ground_truth[doodle_id] = {
                "page_id": page_id,
                "subject": subject,
                "element": element,
            }

            print(f"    -> {element[:60] if element else 'no description'}")

            self._save_ground_truth(ground_truth)
        self._save_manifest(len(ground_truth))

        return SynthStats(
            pages=len(ground_truth) - existing_pairs,
            doodles=len(ground_truth) - existing_pairs,
            output_dir=self.config.output_dir,
        )

    def _setup_dirs(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "pages").mkdir(exist_ok=True)
        (self.config.output_dir / "doodles").mkdir(exist_ok=True)

    def _save_ground_truth(self, gt: dict[str, dict[str, Any]]) -> None:
        path = self.config.output_dir / "ground_truth.json"
        with open(path, "w") as f:
            json.dump(gt, f, indent=2)

    def _save_manifest(self, num_pairs: int) -> None:
        manifest = {
            "version": 2,
            "created_at": datetime.now().isoformat(),
            "seed": self.config.seed,
            "num_pairs": num_pairs,
            "subjects": SUBJECTS,
        }
        path = self.config.output_dir / "manifest.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_existing_ground_truth(self) -> dict[str, dict[str, Any]]:
        gt_path = self.config.output_dir / "ground_truth.json"
        if gt_path.exists():
            with open(gt_path) as f:
                data: dict[str, dict[str, Any]] = json.load(f)
                return data
        return {}

    def _find_next_index(self, ground_truth: dict[str, dict[str, Any]]) -> int:
        if not ground_truth:
            return 0
        indices = [int(k.split("_")[1]) for k in ground_truth.keys()]
        return max(indices) + 1
