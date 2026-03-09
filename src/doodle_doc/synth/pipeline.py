from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
    model: str = "gemini-3.1-flash-image-preview"
    prompt_version: str = "v2"
    concurrency: int = 4
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
        generator_factory: Callable[[], GeminiGenerator] | None = None,
        indexer_factory: Callable[[Path], SynthIndexer] | None = None,
    ) -> None:
        base_config = config or SynthConfig(
            model=settings.synth_model,
            prompt_version=settings.synth_prompt_version,
            concurrency=settings.synth_concurrency,
        )
        self.settings = settings
        self.config = base_config
        self._generator = generator
        self._generator_factory = generator_factory
        self._rng = random.Random(self.config.seed)
        self._indexer_factory = indexer_factory or (
            lambda synth_dir: SynthIndexer(settings, synth_dir)
        )

    def run(self) -> SynthStats:
        self._setup_dirs()

        pages_dir = self.config.output_dir / "pages"
        doodles_dir = self.config.output_dir / "doodles"
        ground_truth = self._load_existing_ground_truth()
        start_idx = self._find_next_index(ground_truth)
        self._advance_rng(start_idx)
        jobs = [
            _SynthJob(
                idx=start_idx + offset,
                archetype=self._rng.choice(PAGE_ARCHETYPES),
            )
            for offset in range(self.config.num_pairs)
        ]
        worker_count = max(1, min(self.config.concurrency, len(jobs) or 1))

        print(f"Generating {len(jobs)} synthetic pairs with concurrency {worker_count}")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    self._generate_pair,
                    job,
                    pages_dir,
                    doodles_dir,
                ): job
                for job in jobs
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(
                    f"[{completed}/{len(jobs)}] Generated {result['page_id']} and {result['doodle_id']}",
                    flush=True,
                )
                ground_truth[result["doodle_id"]] = {
                    "page_id": result["page_id"],
                    "domain": result["domain"],
                    "subject": result["subject"],
                    "archetype": result["archetype"],
                    "element": result["element"],
                }

        self._save_ground_truth(ground_truth)
        self._save_manifest(len(ground_truth))

        print("Indexing synthetic pages with ColQwen2...", flush=True)
        indexer = self._indexer_factory(self.config.output_dir)
        index_stats = indexer.run()

        return SynthStats(
            pages=len(ground_truth),
            doodles=len(ground_truth),
            indexed=index_stats.indexed,
            output_dir=self.config.output_dir,
        )

    def _generator_for_task(self) -> GeminiGenerator:
        if self._generator_factory is not None:
            return self._generator_factory()
        if self._generator is not None:
            return self._generator
        return GeminiGenerator(GeminiConfig(
            model=self.config.model,
            prompt_version=self.config.prompt_version,
        ))

    def _generate_pair(
        self,
        job: _SynthJob,
        pages_dir: Path,
        doodles_dir: Path,
    ) -> dict[str, str]:
        generator = self._generator_for_task()
        page_id = f"page_{job.idx:04d}"
        doodle_id = f"doodle_{job.idx:04d}"

        page = generator.generate_notes_page(job.archetype)
        page.save(pages_dir / f"{page_id}.png", "PNG")

        doodle, element = generator.generate_doodle_for_page(page)
        doodle.save(doodles_dir / f"{doodle_id}.png", "PNG")

        return {
            "page_id": page_id,
            "doodle_id": doodle_id,
            "domain": job.archetype.domain,
            "subject": job.archetype.subject,
            "archetype": job.archetype.key,
            "element": element,
        }

    def _setup_dirs(self) -> None:
        if self.config.clean and self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "pages").mkdir(exist_ok=True)
        (self.config.output_dir / "doodles").mkdir(exist_ok=True)

    def _load_existing_ground_truth(self) -> dict[str, dict[str, Any]]:
        if self.config.clean:
            return {}

        path = self.config.output_dir / "ground_truth.json"
        if not path.exists():
            return {}

        with open(path) as f:
            data: dict[str, dict[str, Any]] = json.load(f)
            return data

    def _find_next_index(self, ground_truth: dict[str, dict[str, Any]]) -> int:
        if not ground_truth:
            return 0

        indices = [
            int(doodle_id.split("_")[1])
            for doodle_id in ground_truth
        ]
        return max(indices) + 1

    def _advance_rng(self, steps: int) -> None:
        for _ in range(steps):
            self._rng.choice(PAGE_ARCHETYPES)

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


@dataclass(frozen=True)
class _SynthJob:
    idx: int
    archetype: Any
