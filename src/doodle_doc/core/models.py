from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    doc_id: str
    path: str
    sha256: str
    modified_time: datetime
    num_pages: int


@dataclass
class Page:
    doc_id: str
    page_num: int
    width_px: int
    height_px: int
    text_layer: str | None = None

@dataclass
class EmbeddingRecord:
    embedding_id: str
    doc_id: str
    page_num: int
    region: str
    model_id: str
    pp_version: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    doc_id: str
    doc_name: str
    page_num: int
    score: float
    thumbnail_url: str
