from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.core.database import Database
from doodle_doc.core.models import SearchResult
from doodle_doc.ingestion.embed import SigLIP2Embedder
from doodle_doc.ingestion.index import FAISSIndex
from doodle_doc.ingestion.preprocess import normalize_sketch
from doodle_doc.search.colqwen_search import ColQwen2SearchService
from doodle_doc.search.fusion import reciprocal_rank_fusion
from doodle_doc.search.rerank import ColQwen2Reranker
from doodle_doc.search.text_search import BM25Index


class SearchService:
    """Stage 1 retrieval service using FAISS."""

    def __init__(
        self,
        settings: Settings,
        embedder: SigLIP2Embedder | None = None,
        index: FAISSIndex | None = None,
        bm25: BM25Index | None = None,
        reranker: ColQwen2Reranker | None = None,
        colqwen_search: ColQwen2SearchService | None = None,
    ) -> None:
        self.settings = settings
        self._embedder = embedder
        self._index = index
        self._bm25 = bm25
        self._reranker = reranker
        self._colqwen_search = colqwen_search
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

    @property
    def reranker(self) -> ColQwen2Reranker:
        if self._reranker is None:
            self._reranker = ColQwen2Reranker(
                model_name=self.settings.colqwen_model,
                batch_size=self.settings.rerank_batch_size,
            )
        return self._reranker

    @property
    def colqwen_search(self) -> ColQwen2SearchService:
        if self._colqwen_search is None:
            self._colqwen_search = ColQwen2SearchService(settings=self.settings)
        return self._colqwen_search

    def search(
        self,
        sketch_image: Image.Image,
        text_query: str | None = None,
        top_k: int | None = None,
        search_mode: str = "fast",
        use_rerank: bool = False,
    ) -> list[SearchResult]:
        """
        Search for pages matching the sketch.

        Args:
            sketch_image: User's sketch as PIL Image
            text_query: Optional text to boost results
            top_k: Number of results to return
            search_mode: "fast" (SigLIP2) or "accurate" (ColQwen2)
            use_rerank: Deprecated, use search_mode instead
        """
        top_k = top_k or self.settings.default_result_k

        if use_rerank and search_mode == "fast":
            search_mode = "accurate"

        if search_mode == "accurate":
            if self.colqwen_search.is_available():
                return self.colqwen_search.search(
                    sketch_image=sketch_image,
                    top_k=top_k,
                )
            # Fall back to old reranking if ColQwen2 index not available
            search_mode = "fast"
            use_rerank = True

        stage1_limit = self.settings.stage1_top_k if use_rerank else top_k

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
            sorted_pages = [key for key, _ in fused[:stage1_limit]]
        else:
            sorted_pages = sorted(page_scores.keys(), key=lambda k: page_scores[k], reverse=True)
            sorted_pages = sorted_pages[:stage1_limit]

        results = []
        for page_key in sorted_pages:
            doc_id, page_num_str = page_key.split(":")
            page_num = int(page_num_str)
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

        if use_rerank and results:
            results = self.reranker.rerank(
                results=results,
                sketch_image=sketch_image,
                rendered_dir=self.settings.rendered_dir,
                top_k=top_k,
            )

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
