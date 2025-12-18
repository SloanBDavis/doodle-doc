from __future__ import annotations

from pydantic import BaseModel


class IngestRequest(BaseModel):
    root_path: str
    force_reindex: bool = False


class IngestResponse(BaseModel):
    job_id: str
    status: str


class IngestStatusResponse(BaseModel):
    status: str
    docs_done: int
    docs_total: int
    pages_done: int
    pages_total: int
    eta_seconds: int | None = None


class SearchResultItem(BaseModel):
    doc_id: str
    doc_name: str
    page_num: int
    score: float
    stage: str
    thumbnail_url: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query_time_ms: int
    total_indexed_pages: int


class HealthResponse(BaseModel):
    status: str
    siglip_loaded: bool
    colqwen_loaded: bool
    indexed_pages: int
    index_size_mb: float


class ModelLoadResponse(BaseModel):
    status: str
    message: str
