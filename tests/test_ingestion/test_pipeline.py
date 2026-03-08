from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

from doodle_doc.core.config import Settings
from doodle_doc.ingestion.pipeline import IngestionPipeline


class FakeColQwenEmbedder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed_batch(self, images: list[object], batch_size: int = 4) -> list[np.ndarray]:
        self.calls.append(len(images))
        return [
            np.full((2, 3), fill_value=i + 1, dtype=np.float32)
            for i, _ in enumerate(images)
        ]


def _make_pdf(path: Path, num_pages: int) -> None:
    doc = fitz.open()
    for page_num in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {page_num}")
    doc.save(path)
    doc.close()


def test_pipeline_batches_colqwen_embeddings_and_indexes_pages(temp_dir: Path) -> None:
    root = temp_dir / "pdfs"
    root.mkdir()
    pdf_path = root / "notes.pdf"
    _make_pdf(pdf_path, num_pages=3)

    settings = Settings(data_dir=temp_dir / "data", colqwen_batch_size=2)
    embedder = FakeColQwenEmbedder()
    pipeline = IngestionPipeline(settings=settings, colqwen_embedder=embedder)  # type: ignore[arg-type]

    progress = pipeline.run(root)

    assert progress.pages_done == 3
    assert embedder.calls == [2, 1]
    assert pipeline.colqwen_index.page_count == 3
    assert len(pipeline.db.get_all_documents()) == 1


def test_pipeline_replaces_existing_document_by_path(temp_dir: Path) -> None:
    root = temp_dir / "pdfs"
    root.mkdir()
    pdf_path = root / "notes.pdf"
    _make_pdf(pdf_path, num_pages=2)

    settings = Settings(data_dir=temp_dir / "data", colqwen_batch_size=2)
    embedder = FakeColQwenEmbedder()
    pipeline = IngestionPipeline(settings=settings, colqwen_embedder=embedder)  # type: ignore[arg-type]

    pipeline.run(root)
    _make_pdf(pdf_path, num_pages=4)
    pipeline.run(root)

    docs = pipeline.db.get_all_documents()
    assert len(docs) == 1
    assert docs[0].num_pages == 4
    assert pipeline.colqwen_index.page_count == 4
