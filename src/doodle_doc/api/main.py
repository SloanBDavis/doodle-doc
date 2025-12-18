from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from doodle_doc.api.routes import (
    health_router,
    ingest_router,
    search_router,
    documents_router,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="DoodleDoc API",
        description="Sketch-based search for handwritten PDF notes",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(search_router)
    app.include_router(documents_router)

    return app


app = create_app()
