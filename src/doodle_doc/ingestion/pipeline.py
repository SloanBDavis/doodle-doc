from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
import numpy as np

from doodle_doc.core.config import Settings
from doodle_doc.core.database import Database, DocumentModel, PageModel
from doodle_doc.ingestion.colqwen_embed import ColQwen2Embedder
from doodle_doc.ingestion.colqwen_index import ColQwen2Index
from doodle_doc.ingestion.discover import PDFFile, discover_pdfs, filter_unchanged
from doodle_doc.ingestion.embed import SigLIP2Embedder
from doodle_doc.ingestion.index import FAISSIndex
from doodle_doc.ingestion.preprocess import normalize_ink
from doodle_doc.ingestion.regions import extract_regions
from doodle_doc.ingestion.render import render_page, extract_text_layer, get_page_count


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
        embedder: SigLIP2Embedder | None = None,
        colqwen_embedder: ColQwen2Embedder | None = None,
    ) -> None:
        self.settings = settings
        self._embedder = embedder
        self._colqwen_embedder = colqwen_embedder
        self._index: FAISSIndex | None = None
        self._colqwen_index: ColQwen2Index | None = None
        self._db: Database | None = None

    @property
    def embedder(self) -> SigLIP2Embedder:
        if self._embedder is None:
            self._embedder = SigLIP2Embedder(
                model_name=self.settings.siglip_model,
            )
        return self._embedder

    @property
    def index(self) -> FAISSIndex:
        if self._index is None:
            index_path = self.settings.index_dir
            if (index_path / "faiss.index").exists():
                self._index = FAISSIndex.load(index_path)
            else:
                self._index = FAISSIndex(self.settings.embedding_dim)
        return self._index

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

        if not force_reindex:
            existing = {doc.sha256 for doc in self.db.get_all_documents()}
            pdfs = filter_unchanged(pdfs, existing)

        progress.docs_total = len(pdfs)
        progress.pages_total = sum(get_page_count(str(p.path)) for p in pdfs)
        progress.status = "indexing"
        self._notify(on_progress, progress)

        for pdf in pdfs:
            progress.current_doc = pdf.path.name
            self._notify(on_progress, progress)

            self._process_pdf(pdf, progress, on_progress)
            progress.docs_done += 1
            self._notify(on_progress, progress)

        self.index.save(self.settings.index_dir)

        if self.settings.colqwen_index_enabled:
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
        num_pages = get_page_count(str(pdf.path))
        num_pages = min(num_pages, self.settings.max_pages_per_doc)

        doc = DocumentModel(
            doc_id=doc_id,
            path=str(pdf.path),
            sha256=pdf.sha256,
            modified_time=datetime.fromtimestamp(pdf.path.stat().st_mtime),
            num_pages=num_pages,
        )
        self.db.add_document(doc)

        rendered_dir = self.settings.rendered_dir / doc_id
        rendered_dir.mkdir(parents=True, exist_ok=True)

        for page_num in range(num_pages):
            img = render_page(str(pdf.path), page_num, self.settings.render_dpi)

            img_path = rendered_dir / f"{page_num}.png"
            img.save(img_path)

            text_layer = extract_text_layer(str(pdf.path), page_num)

            page = PageModel(
                doc_id=doc_id,
                page_num=page_num,
                width_px=img.width,
                height_px=img.height,
                text_layer=text_layer,
            )
            self.db.add_page(page)

            normalized = normalize_ink(
                img,
                self.settings.clahe_clip_limit,
                self.settings.clahe_grid_size,
            )
            regions = extract_regions(normalized)

            embeddings = []
            metadata = []
            for region_name, region_img in regions.items():
                emb = self.embedder.embed_single(region_img)
                embeddings.append(emb)
                metadata.append({
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "region": region_name,
                })

            self.index.add(np.array(embeddings), metadata)

            if self.settings.colqwen_index_enabled:
                colqwen_emb = self.colqwen_embedder.embed_single(img)
                self.colqwen_index.add(doc_id, page_num, colqwen_emb)

            progress.pages_done += 1
            self._notify(on_progress, progress)

    def _notify(
        self,
        callback: ProgressCallback | None,
        progress: IndexingProgress,
    ) -> None:
        if callback is not None:
            callback(progress)
