from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


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


RegionType = Literal["full", "q1", "q2", "q3", "q4"]


@dataclass
class EmbeddingRecord:
    embedding_id: str
    doc_id: str
    page_num: int
    region: RegionType
    model_id: str
    pp_version: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    doc_id: str
    doc_name: str
    page_num: int
    score: float
    stage: Literal["fast", "reranked"]
    thumbnail_url: str
