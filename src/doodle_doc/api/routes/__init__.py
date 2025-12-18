from __future__ import annotations

from doodle_doc.api.routes.health import router as health_router
from doodle_doc.api.routes.ingest import router as ingest_router
from doodle_doc.api.routes.search import router as search_router
from doodle_doc.api.routes.documents import router as documents_router

__all__ = ["health_router", "ingest_router", "search_router", "documents_router"]
