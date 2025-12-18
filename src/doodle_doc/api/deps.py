from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from doodle_doc.core.config import Settings, get_settings
from doodle_doc.core.database import Database
from doodle_doc.ingestion.embed import SigLIP2Embedder
from doodle_doc.ingestion.index import FAISSIndex
from doodle_doc.search.retrieval import SearchService
from doodle_doc.search.rerank import ColQwen2Reranker

if TYPE_CHECKING:
    pass


class AppState:
    """Application state holding shared resources."""

    def __init__(self) -> None:
        self._settings: Settings | None = None
        self._embedder: SigLIP2Embedder | None = None
        self._index: FAISSIndex | None = None
        self._db: Database | None = None
        self._search_service: SearchService | None = None
        self._reranker: ColQwen2Reranker | None = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def embedder(self) -> SigLIP2Embedder:
        if self._embedder is None:
            self._embedder = SigLIP2Embedder(model_name=self.settings.siglip_model)
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
            self._db = Database(self.settings.index_dir / "metadata.sqlite")
        return self._db

    @property
    def search_service(self) -> SearchService:
        if self._search_service is None:
            self._search_service = SearchService(
                settings=self.settings,
                embedder=self._embedder,
                index=self._index,
            )
        return self._search_service

    @property
    def reranker(self) -> ColQwen2Reranker:
        if self._reranker is None:
            self._reranker = ColQwen2Reranker(model_name=self.settings.colqwen_model)
        return self._reranker

    def is_embedder_loaded(self) -> bool:
        return self._embedder is not None

    def is_reranker_loaded(self) -> bool:
        return self._reranker is not None and self._reranker.is_loaded()


@lru_cache
def get_app_state() -> AppState:
    return AppState()
