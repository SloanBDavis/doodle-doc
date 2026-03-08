from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from doodle_doc.core.config import Settings
from doodle_doc.synth.gemini_generator import (
    PAGE_ARCHETYPES,
    GeminiConfig,
    GeminiGenerator,
)
from doodle_doc.synth.synth_index import SynthIndexer


@dataclass
class SynthConfig:
    output_dir: Path = Path("data/synth")
    num_pairs: int = 25
    seed: int = 42
    model: str = "gemini-2.5-flash-image"
    prompt_version: str = "v2"
    clean: bool = True


@dataclass
class SynthStats:
    pages: int
    doodles: int
    indexed: int
    output_dir: Path


class SynthPipeline:
    def __init__(
        self,
        settings: Settings,
        config: SynthConfig | None = None,
        generator: GeminiGenerator | None = None,
        indexer_factory: Callable[[Path], SynthIndexer] | None = None,
    ) -> None:
        base_config = config or SynthConfig(
            model=settings.synth_model,
            prompt_version=settings.synth_prompt_version,
        )
        self.settings = settings
        self.config = base_config
        self._generator = generator
        self._rng = random.Random(self.config.seed)
        self._indexer_factory = indexer_factory or (
            lambda synth_dir: SynthIndexer(settings, synth_dir)
        )

    @property
    def generator(self) -> GeminiGenerator:
        if self._generator is None:
            self._generator = GeminiGenerator(GeminiConfig(
                model=self.config.model,
                prompt_version=self.config.prompt_version,
            ))
        return self._generator

    def run(self) -> SynthStats:
        self._setup_dirs()

        pages_dir = self.config.output_dir / "pages"
        doodles_dir = self.config.output_dir / "doodles"
        ground_truth: dict[str, dict[str, Any]] = {}

        for idx in range(self.config.num_pairs):
            archetype = self._rng.choice(PAGE_ARCHETYPES)
            page_id = f"page_{idx:04d}"
            doodle_id = f"doodle_{idx:04d}"

            page = self.generator.generate_notes_page(archetype)
            page.save(pages_dir / f"{page_id}.png", "PNG")

            doodle, element = self.generator.generate_doodle_for_page(page)
            doodle.save(doodles_dir / f"{doodle_id}.png", "PNG")

            ground_truth[doodle_id] = {
                "page_id": page_id,
                "domain": archetype.domain,
                "subject": archetype.subject,
                "archetype": archetype.key,
                "element": element,
            }

        self._save_ground_truth(ground_truth)
        self._save_manifest(len(ground_truth))

        indexer = self._indexer_factory(self.config.output_dir)
        index_stats = indexer.run()

        return SynthStats(
            pages=len(ground_truth),
            doodles=len(ground_truth),
            indexed=index_stats.indexed,
            output_dir=self.config.output_dir,
        )

    def _setup_dirs(self) -> None:
        if self.config.clean and self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "pages").mkdir(exist_ok=True)
        (self.config.output_dir / "doodles").mkdir(exist_ok=True)

    def _save_ground_truth(self, gt: dict[str, dict[str, Any]]) -> None:
        path = self.config.output_dir / "ground_truth.json"
        with open(path, "w") as f:
            json.dump(gt, f, indent=2)

    def _save_manifest(self, num_pairs: int) -> None:
        manifest = {
            "version": 3,
            "created_at": datetime.now().isoformat(),
            "seed": self.config.seed,
            "num_pairs": num_pairs,
            "model": self.config.model,
            "prompt_version": self.config.prompt_version,
            "output_dir": str(self.config.output_dir),
            "archetypes": [asdict(archetype) for archetype in PAGE_ARCHETYPES],
            "files": {
                "pages": "pages/page_####.png",
                "doodles": "doodles/doodle_####.png",
                "ground_truth": "ground_truth.json",
                "index": "index/colqwen",
            },
        }
        path = self.config.output_dir / "manifest.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
