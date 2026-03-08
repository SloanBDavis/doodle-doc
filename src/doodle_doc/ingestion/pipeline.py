from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import fitz
from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.core.database import Database, DocumentModel, PageModel
from doodle_doc.ingestion.colqwen_embed import ColQwen2Embedder
from doodle_doc.ingestion.colqwen_index import ColQwen2Index
from doodle_doc.ingestion.discover import PDFFile, discover_pdfs, filter_unchanged
from doodle_doc.ingestion.render import get_page_count


@dataclass
class IndexingProgress:
    docs_done: int = 0
    docs_total: int = 0
    pages_done: int = 0
    pages_total: int = 0
    current_doc: str = ""
    status: str = "pending"


@dataclass
class IndexingJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    progress: IndexingProgress = field(default_factory=IndexingProgress)


ProgressCallback = Callable[[IndexingProgress], None]


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        colqwen_embedder: ColQwen2Embedder | None = None,
        colqwen_index: ColQwen2Index | None = None,
        db: Database | None = None,
    ) -> None:
        self.settings = settings
        self._colqwen_embedder = colqwen_embedder
        self._colqwen_index = colqwen_index
        self._db = db

    @property
    def db(self) -> Database:
        if self._db is None:
            db_path = self.settings.index_dir / "metadata.sqlite"
            self._db = Database(db_path)
        return self._db

    @property
    def colqwen_embedder(self) -> ColQwen2Embedder:
        if self._colqwen_embedder is None:
            self._colqwen_embedder = ColQwen2Embedder(
                model_name=self.settings.colqwen_model,
            )
        return self._colqwen_embedder

    @property
    def colqwen_index(self) -> ColQwen2Index:
        if self._colqwen_index is None:
            colqwen_dir = self.settings.colqwen_index_dir
            if (colqwen_dir / "manifest.json").exists():
                self._colqwen_index = ColQwen2Index.load(colqwen_dir)
            else:
                self._colqwen_index = ColQwen2Index(colqwen_dir)
                self._colqwen_index.set_model(self.settings.colqwen_model)
        return self._colqwen_index

    def run(
        self,
        root: Path,
        on_progress: ProgressCallback | None = None,
        force_reindex: bool = False,
    ) -> IndexingProgress:
        """Run the full ingestion pipeline."""
        progress = IndexingProgress(status="discovering")
        self._notify(on_progress, progress)

        pdfs = discover_pdfs(root)
        existing_docs = self.db.get_all_documents()
        docs_by_path = {doc.path: doc for doc in existing_docs}

        if not force_reindex:
            existing_hashes = {doc.sha256 for doc in existing_docs}
            pdfs = filter_unchanged(pdfs, existing_hashes)

        progress.docs_total = len(pdfs)
        progress.pages_total = sum(
            min(get_page_count(str(pdf.path)), self.settings.max_pages_per_doc)
            for pdf in pdfs
        )
        progress.status = "indexing"
        self._notify(on_progress, progress)

        for pdf in pdfs:
            progress.current_doc = pdf.path.name
            self._notify(on_progress, progress)

            existing_doc = docs_by_path.get(str(pdf.path))
            if existing_doc is not None:
                self.remove_document(existing_doc.doc_id)

            self._process_pdf(pdf, progress, on_progress)
            progress.docs_done += 1
            self._notify(on_progress, progress)

        self.colqwen_index.save()

        progress.status = "completed"
        self._notify(on_progress, progress)
        return progress

    def _process_pdf(
        self,
        pdf: PDFFile,
        progress: IndexingProgress,
        on_progress: ProgressCallback | None,
    ) -> None:
        """Process a single PDF file."""
        doc_id = str(uuid.uuid4())
        with fitz.open(str(pdf.path)) as doc_handle:
            num_pages = min(len(doc_handle), self.settings.max_pages_per_doc)
            self.db.add_document(DocumentModel(
                doc_id=doc_id,
                path=str(pdf.path),
                sha256=pdf.sha256,
                modified_time=datetime.fromtimestamp(pdf.path.stat().st_mtime),
                num_pages=num_pages,
            ))

            rendered_dir = self.settings.rendered_dir / doc_id
            rendered_dir.mkdir(parents=True, exist_ok=True)

            batch_size = self.settings.colqwen_batch_size
            for batch_start in range(0, num_pages, batch_size):
                page_batch = list(range(batch_start, min(batch_start + batch_size, num_pages)))
                page_models: list[PageModel] = []
                images: list[Image.Image] = []

                for page_num in page_batch:
                    page_image, text_layer = self._load_page(doc_handle, page_num)
                    images.append(page_image)
                    page_image.save(rendered_dir / f"{page_num}.png")
                    page_models.append(PageModel(
                        doc_id=doc_id,
                        page_num=page_num,
                        width_px=page_image.width,
                        height_px=page_image.height,
                        text_layer=text_layer,
                    ))

                self.db.add_pages(page_models)
                embeddings = self.colqwen_embedder.embed_batch(
                    images,
                    batch_size=self.settings.colqwen_batch_size,
                )
                self.colqwen_index.add_many([
                    (doc_id, page_num, embedding)
                    for page_num, embedding in zip(page_batch, embeddings, strict=True)
                ])

                progress.pages_done += len(page_batch)
                self._notify(on_progress, progress)

    def remove_document(self, doc_id: str) -> None:
        self.db.delete_document(doc_id)
        self.colqwen_index.remove_by_doc_id(doc_id)
        rendered_dir = self.settings.rendered_dir / doc_id
        if rendered_dir.exists():
            shutil.rmtree(rendered_dir)

    def _load_page(
        self,
        doc_handle: fitz.Document,
        page_num: int,
    ) -> tuple[Image.Image, str | None]:
        page = doc_handle[page_num]
        zoom = self.settings.render_dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        text_layer = page.get_text("text").strip()
        return image, text_layer or None

    def _notify(
        self,
        callback: ProgressCallback | None,
        progress: IndexingProgress,
    ) -> None:
        if callback is not None:
            callback(progress)
