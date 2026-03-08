from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.synth.pipeline import SynthConfig, SynthPipeline
from doodle_doc.synth.synth_index import SynthIndexStats


class FakeGenerator:
    def generate_notes_page(self, archetype: object) -> Image.Image:
        return Image.new("RGB", (128, 128), "white")

    def generate_doodle_for_page(self, page_image: Image.Image) -> tuple[Image.Image, str]:
        return Image.new("RGB", (64, 64), "white"), "triangle doodle"


class FakeIndexer:
    def __init__(self, synth_dir: Path) -> None:
        self.synth_dir = synth_dir

    def run(self) -> SynthIndexStats:
        index_dir = self.synth_dir / "index" / "colqwen"
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "manifest.json").write_text("{}")
        return SynthIndexStats(total_pages=3, indexed=3, skipped=0)


def test_synth_pipeline_builds_eval_ready_dataset(temp_dir: Path) -> None:
    settings = Settings(data_dir=temp_dir / "data")
    output_dir = temp_dir / "synth"
    config = SynthConfig(
        output_dir=output_dir,
        num_pairs=3,
        seed=7,
        model="test-model",
        prompt_version="v-test",
    )
    pipeline = SynthPipeline(
        settings=settings,
        config=config,
        generator=FakeGenerator(),  # type: ignore[arg-type]
        indexer_factory=lambda synth_dir: FakeIndexer(synth_dir),  # type: ignore[arg-type]
    )

    stats = pipeline.run()

    assert stats.pages == 3
    assert stats.doodles == 3
    assert stats.indexed == 3
    assert (output_dir / "index" / "colqwen" / "manifest.json").exists()

    with open(output_dir / "manifest.json") as f:
        manifest = json.load(f)
    with open(output_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    assert manifest["model"] == "test-model"
    assert manifest["prompt_version"] == "v-test"
    assert len(ground_truth) == 3


def test_synth_pipeline_cleans_output_dir_by_default(temp_dir: Path) -> None:
    settings = Settings(data_dir=temp_dir / "data")
    output_dir = temp_dir / "synth"
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_file = output_dir / "stale.txt"
    stale_file.write_text("old")

    pipeline = SynthPipeline(
        settings=settings,
        config=SynthConfig(output_dir=output_dir, num_pairs=1),
        generator=FakeGenerator(),  # type: ignore[arg-type]
        indexer_factory=lambda synth_dir: FakeIndexer(synth_dir),  # type: ignore[arg-type]
    )

    pipeline.run()

    assert not stale_file.exists()
