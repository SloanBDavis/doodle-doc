from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from doodle_doc.api.deps import AppState, get_app_state

router = APIRouter(prefix="/v1", tags=["documents"])


@router.get("/doc/{doc_id}/page/{page_num}")
def get_page(
    doc_id: str,
    page_num: int,
    state: AppState = Depends(get_app_state),
) -> FileResponse:
    """Get full page image for viewer."""
    image_path = state.settings.rendered_dir / doc_id / f"{page_num}.png"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(image_path, media_type="image/png")


@router.get("/thumb/{doc_id}/{page_num}")
def get_thumbnail(
    doc_id: str,
    page_num: int,
    state: AppState = Depends(get_app_state),
) -> FileResponse:
    """Get page thumbnail (uses same image, browser will resize)."""
    # TODO: Generate actual thumbnails at 300px width
    image_path = state.settings.rendered_dir / doc_id / f"{page_num}.png"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(image_path, media_type="image/png")


@router.get("/documents")
def list_documents(
    state: AppState = Depends(get_app_state),
) -> list[dict]:
    """List all indexed documents."""
    docs = state.db.get_all_documents()
    return [
        {
            "doc_id": doc.doc_id,
            "path": doc.path,
            "name": Path(doc.path).name,
            "num_pages": doc.num_pages,
            "sha256": doc.sha256,
        }
        for doc in docs
    ]
