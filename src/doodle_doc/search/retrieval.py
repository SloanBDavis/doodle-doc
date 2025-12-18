from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.core.database import Database
from doodle_doc.core.models import SearchResult
from doodle_doc.ingestion.embed import SigLIP2Embedder
from doodle_doc.ingestion.index import FAISSIndex
from doodle_doc.ingestion.preprocess import normalize_sketch
from doodle_doc.search.fusion import reciprocal_rank_fusion
from doodle_doc.search.text_search import BM25Index


class SearchService:
    """Stage 1 retrieval service using FAISS."""

    def __init__(
        self,
        settings: Settings,
        embedder: SigLIP2Embedder | None = None,
        index: FAISSIndex | None = None,
        bm25: BM25Index | None = None,
    ) -> None:
        self.settings = settings
        self._embedder = embedder
        self._index = index
        self._bm25 = bm25
        self._db: Database | None = None

    @property
    def embedder(self) -> SigLIP2Embedder:
        if self._embedder is None:
            self._embedder = SigLIP2Embedder(model_name=self.settings.siglip_model)
        return self._embedder

    @property
    def index(self) -> FAISSIndex:
        if self._index is None:
            self._index = FAISSIndex.load(self.settings.index_dir)
        return self._index

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.load(self.settings.index_dir / "bm25")
        return self._bm25

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = Database(self.settings.index_dir / "metadata.sqlite")
        return self._db

    def search(
        self,
        sketch_image: Image.Image,
        text_query: str | None = None,
        top_k: int | None = None,
        use_rerank: bool = False,
    ) -> list[SearchResult]:
        """
        Search for pages matching the sketch.

        Args:
            sketch_image: User's sketch as PIL Image
            text_query: Optional text to boost results
            top_k: Number of results to return
            use_rerank: Whether to use Stage 2 reranking (not yet implemented)
        """
        top_k = top_k or self.settings.default_result_k

        normalized = normalize_sketch(
            sketch_image,
            self.settings.clahe_clip_limit,
            self.settings.clahe_grid_size,
        )

        query_embedding = self.embedder.embed_single(normalized)

        raw_results = self.index.search(query_embedding, self.settings.stage1_top_k)

        page_scores = self._aggregate_by_page(raw_results)

        if text_query and self.settings.enable_text_boost:
            text_results = self.bm25.search(text_query, self.settings.stage1_top_k)
            text_page_scores = [(f"{m['doc_id']}:{m['page_num']}", s) for m, s in text_results]
            visual_page_scores = [(k, s) for k, s in page_scores.items()]

            fused = reciprocal_rank_fusion([visual_page_scores, text_page_scores])
            sorted_pages = [key for key, _ in fused[:top_k]]
        else:
            sorted_pages = sorted(page_scores.keys(), key=lambda k: page_scores[k], reverse=True)
            sorted_pages = sorted_pages[:top_k]

        results = []
        for page_key in sorted_pages:
            doc_id, page_num = page_key.split(":")
            page_num = int(page_num)
            doc = self.db.get_document(doc_id)
            if doc:
                results.append(SearchResult(
                    doc_id=doc_id,
                    doc_name=Path(doc.path).name,
                    page_num=page_num,
                    score=page_scores.get(page_key, 0.0),
                    stage="fast",
                    thumbnail_url=f"/v1/thumb/{doc_id}/{page_num}",
                ))

        return results

    def _aggregate_by_page(
        self,
        results: list[tuple[dict[str, Any], float]],
    ) -> dict[str, float]:
        """Aggregate region scores to page level (max score per page)."""
        page_scores: dict[str, float] = defaultdict(float)
        for meta, score in results:
            key = f"{meta['doc_id']}:{meta['page_num']}"
            page_scores[key] = max(page_scores[key], score)
        return page_scores
