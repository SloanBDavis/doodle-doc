from __future__ import annotations

from functools import lru_cache

from doodle_doc.core.config import Settings, get_settings
from doodle_doc.core.database import Database
from doodle_doc.ingestion.colqwen_embed import ColQwen2Embedder
from doodle_doc.ingestion.colqwen_index import ColQwen2Index
from doodle_doc.search.retrieval import SearchService


class AppState:
    """Application state holding shared resources."""

    def __init__(self) -> None:
        self._settings: Settings | None = None
        self._colqwen_embedder: ColQwen2Embedder | None = None
        self._colqwen_index: ColQwen2Index | None = None
        self._db: Database | None = None
        self._search_service: SearchService | None = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def colqwen_embedder(self) -> ColQwen2Embedder:
        if self._colqwen_embedder is None:
            self._colqwen_embedder = ColQwen2Embedder(model_name=self.settings.colqwen_model)
        return self._colqwen_embedder

    @property
    def colqwen_index(self) -> ColQwen2Index:
        if self._colqwen_index is None:
            self._colqwen_index = ColQwen2Index.load(self.settings.colqwen_index_dir)
        return self._colqwen_index

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
                embedder=self.colqwen_embedder,
                index=self.colqwen_index,
                db=self.db,
            )
        return self._search_service

    def is_colqwen_loaded(self) -> bool:
        return self.colqwen_embedder.is_loaded()


@lru_cache
def get_app_state() -> AppState:
    return AppState()
