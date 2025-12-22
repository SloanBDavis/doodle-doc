from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from doodle_doc.api.deps import AppState, get_app_state

router = APIRouter(prefix="/v1", tags=["documents"])


class DocumentIdsRequest(BaseModel):
    doc_ids: list[str]


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


@router.delete("/documents")
def remove_documents(
    request: DocumentIdsRequest,
    state: AppState = Depends(get_app_state),
) -> dict:
    """Remove documents from index."""
    for doc_id in request.doc_ids:
        # Remove from database
        state.db.delete_document(doc_id)
        # Remove from FAISS index
        state.index.remove_by_doc_id(doc_id)
        # Remove rendered images
        rendered_dir = state.settings.rendered_dir / doc_id
        if rendered_dir.exists():
            shutil.rmtree(rendered_dir)

    # Save updated index
    state.index.save(state.settings.index_dir)

    return {"removed": len(request.doc_ids)}


@router.post("/documents/reindex")
def reindex_documents(
    request: DocumentIdsRequest,
    state: AppState = Depends(get_app_state),
) -> dict:
    """Re-index specific documents by their doc_ids."""
    from doodle_doc.ingestion.discover import PDFFile
    from doodle_doc.ingestion.pipeline import IndexingProgress, IngestionPipeline

    reindexed = 0
    for doc_id in request.doc_ids:
        doc = state.db.get_document(doc_id)
        if not doc:
            continue

        pdf_path = Path(doc.path)
        if not pdf_path.exists():
            continue

        # Remove old data first
        state.db.delete_document(doc_id)
        state.index.remove_by_doc_id(doc_id)
        rendered_dir = state.settings.rendered_dir / doc_id
        if rendered_dir.exists():
            shutil.rmtree(rendered_dir)

        # Re-index the document
        sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        pdf_file = PDFFile(path=pdf_path, sha256=sha256, size_bytes=pdf_path.stat().st_size)

        pipeline = IngestionPipeline(
            settings=state.settings,
            embedder=state.embedder if state.is_embedder_loaded() else None,
        )
        pipeline._index = state.index
        pipeline._db = state.db

        # Create a dummy progress object
        progress = IndexingProgress()
        pipeline._process_pdf(pdf_file, progress, None)
        reindexed += 1

    state.index.save(state.settings.index_dir)
    return {"reindexed": reindexed}
