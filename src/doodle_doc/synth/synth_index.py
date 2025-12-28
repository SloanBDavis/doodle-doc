from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.ingestion.colqwen_embed import ColQwen2Embedder
from doodle_doc.ingestion.colqwen_index import ColQwen2Index


@dataclass
class SynthIndexStats:
    total_pages: int
    indexed: int
    skipped: int


class SynthIndexer:
    """Index synthetic pages using ColQwen2 embeddings."""

    def __init__(self, settings: Settings, synth_dir: Path) -> None:
        self.settings = settings
        self.synth_dir = synth_dir
        self.pages_dir = synth_dir / "pages"
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
            if (self.index_dir / "manifest.json").exists():
                self._index = ColQwen2Index.load(self.index_dir)
            else:
                self._index = ColQwen2Index(self.index_dir)
                self._index.set_model(self.settings.colqwen_model)
        return self._index

    def run(self) -> SynthIndexStats:
        page_files = sorted(self.pages_dir.glob("*.png"))

        if not page_files:
            print(f"No pages found in {self.pages_dir}")
            return SynthIndexStats(total_pages=0, indexed=0, skipped=0)

        indexed = 0
        skipped = 0

        for i, page_path in enumerate(page_files):
            doc_id = page_path.stem  # e.g., "page_0000"
            page_num = 0  # Synth pages are single images

            if self.index.has_page(doc_id, page_num):
                print(f"[{i+1}/{len(page_files)}] {doc_id} (skipped, already indexed)")
                skipped += 1
                continue

            print(f"[{i+1}/{len(page_files)}] Indexing {doc_id}...")

            img = Image.open(page_path)
            embedding = self.embedder.embed_single(img)

            self.index.add(doc_id, page_num, embedding)
            self.index.save()
            indexed += 1

        return SynthIndexStats(
            total_pages=len(page_files),
            indexed=indexed,
            skipped=skipped,
        )
