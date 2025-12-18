from __future__ import annotations

from fastapi import APIRouter, Depends

from doodle_doc.api.deps import AppState, get_app_state
from doodle_doc.api.schemas import HealthResponse, ModelLoadResponse

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health(state: AppState = Depends(get_app_state)) -> HealthResponse:
    index_size_mb = 0.0
    index_path = state.settings.index_dir / "faiss.index"
    if index_path.exists():
        index_size_mb = index_path.stat().st_size / (1024 * 1024)

    return HealthResponse(
        status="ok",
        siglip_loaded=state.is_embedder_loaded(),
        colqwen_loaded=state.is_reranker_loaded(),
        indexed_pages=state.index.size // 5 if state._index else 0,
        index_size_mb=round(index_size_mb, 2),
    )


@router.post("/models/colqwen/load", response_model=ModelLoadResponse)
def load_colqwen(state: AppState = Depends(get_app_state)) -> ModelLoadResponse:
    state.reranker.load()
    return ModelLoadResponse(status="ok", message="ColQwen2 model loaded")


@router.post("/models/colqwen/unload", response_model=ModelLoadResponse)
def unload_colqwen(state: AppState = Depends(get_app_state)) -> ModelLoadResponse:
    state.reranker.unload()
    return ModelLoadResponse(status="ok", message="ColQwen2 model unloaded")
