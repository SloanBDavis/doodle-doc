from __future__ import annotations

from doodle_doc.ingestion.render import render_page, extract_text_layer
from doodle_doc.ingestion.preprocess import normalize_ink, resize_with_padding
from doodle_doc.ingestion.regions import extract_regions
from doodle_doc.ingestion.embed import SigLIP2Embedder
from doodle_doc.ingestion.index import FAISSIndex

__all__ = [
    "render_page",
    "extract_text_layer",
    "normalize_ink",
    "resize_with_padding",
    "extract_regions",
    "SigLIP2Embedder",
    "FAISSIndex",
]
