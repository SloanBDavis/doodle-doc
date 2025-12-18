from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from doodle_doc.api.deps import AppState, get_app_state
from doodle_doc.api.schemas import IngestRequest, IngestResponse, IngestStatusResponse
from doodle_doc.ingestion.pipeline import IngestionPipeline, IndexingProgress

router = APIRouter(prefix="/v1", tags=["ingest"])

_jobs: dict[str, IndexingProgress] = {}
_job_threads: dict[str, threading.Thread] = {}


def _run_indexing(
    job_id: str,
    root_path: Path,
    force_reindex: bool,
    state: AppState,
) -> None:
    def on_progress(progress: IndexingProgress) -> None:
        _jobs[job_id] = progress

    pipeline = IngestionPipeline(
        settings=state.settings,
        embedder=state._embedder,
    )
    pipeline.run(root_path, on_progress=on_progress, force_reindex=force_reindex)


@router.post("/ingest", response_model=IngestResponse)
def start_ingest(
    request: IngestRequest,
    state: AppState = Depends(get_app_state),
) -> IngestResponse:
    root_path = Path(request.root_path)
    if not root_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {root_path}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = IndexingProgress(status="starting")

    thread = threading.Thread(
        target=_run_indexing,
        args=(job_id, root_path, request.force_reindex, state),
        daemon=True,
    )
    _job_threads[job_id] = thread
    thread.start()

    return IngestResponse(job_id=job_id, status="started")


@router.get("/ingest/{job_id}", response_model=IngestStatusResponse)
def get_ingest_status(job_id: str) -> IngestStatusResponse:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    progress = _jobs[job_id]

    eta = None
    if progress.pages_done > 0 and progress.pages_total > progress.pages_done:
        # Rough estimate based on progress
        remaining = progress.pages_total - progress.pages_done
        eta = int(remaining * 0.5)  # ~0.5s per page estimate

    return IngestStatusResponse(
        status=progress.status,
        docs_done=progress.docs_done,
        docs_total=progress.docs_total,
        pages_done=progress.pages_done,
        pages_total=progress.pages_total,
        eta_seconds=eta,
    )
