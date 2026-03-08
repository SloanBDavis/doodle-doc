from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    def __init__(
        self,
        settings: Settings,
        synth_dir: Path,
        embedder: ColQwen2Embedder | None = None,
        index: ColQwen2Index | None = None,
    ) -> None:
        self.settings = settings
        self.synth_dir = synth_dir
        self.pages_dir = synth_dir / "pages"
        self.index_dir = synth_dir / "index" / "colqwen"
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
            if (self.index_dir / "manifest.json").exists():
                self._index = ColQwen2Index.load(self.index_dir)
            else:
                self._index = ColQwen2Index(self.index_dir)
                self._index.set_model(self.settings.colqwen_model)
        return self._index

    def run(self) -> SynthIndexStats:
        page_files = sorted(self.pages_dir.glob("*.png"))
        if not page_files:
            return SynthIndexStats(total_pages=0, indexed=0, skipped=0)

        indexed = 0
        skipped = 0
        batch_size = self.settings.colqwen_batch_size

        for batch_start in range(0, len(page_files), batch_size):
            file_batch = page_files[batch_start : batch_start + batch_size]
            images = [Image.open(page_path).convert("RGB") for page_path in file_batch]
            embeddings = self.embedder.embed_batch(images, batch_size=batch_size)
            records: list[tuple[str, int, Any]] = []

            for page_path, embedding in zip(file_batch, embeddings, strict=True):
                doc_id = page_path.stem
                page_num = 0
                if self.index.has_page(doc_id, page_num):
                    skipped += 1
                    continue
                records.append((doc_id, page_num, embedding))
                indexed += 1

            if records:
                self.index.add_many(records)

        self.index.save()
        return SynthIndexStats(
            total_pages=len(page_files),
            indexed=indexed,
            skipped=skipped,
        )
