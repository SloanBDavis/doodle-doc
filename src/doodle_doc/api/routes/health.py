from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from doodle_doc.api.deps import AppState, get_app_state
from doodle_doc.api.schemas import HealthResponse, ModelLoadResponse

router = APIRouter(prefix="/v1", tags=["health"])


def _directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0

    total_bytes = sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if entry.is_file()
    )
    return total_bytes / (1024 * 1024)


@router.get("/health", response_model=HealthResponse)
def get_health(state: AppState = Depends(get_app_state)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        colqwen_loaded=state.is_colqwen_loaded(),
        indexed_pages=state.colqwen_index.page_count,
        index_size_mb=round(_directory_size_mb(state.settings.colqwen_index_dir), 2),
    )


@router.post("/models/colqwen/load", response_model=ModelLoadResponse)
def load_colqwen(state: AppState = Depends(get_app_state)) -> ModelLoadResponse:
    state.colqwen_embedder.load()
    return ModelLoadResponse(status="ok", message="ColQwen2 model loaded")


@router.post("/models/colqwen/unload", response_model=ModelLoadResponse)
def unload_colqwen(state: AppState = Depends(get_app_state)) -> ModelLoadResponse:
    state.colqwen_embedder.unload()
    return ModelLoadResponse(status="ok", message="ColQwen2 model unloaded")
