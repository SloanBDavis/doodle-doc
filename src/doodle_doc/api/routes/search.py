from __future__ import annotations

import time
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, UploadFile
from PIL import Image

from doodle_doc.api.deps import AppState, get_app_state
from doodle_doc.api.schemas import SearchResponse, SearchResultItem

router = APIRouter(prefix="/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    sketch_image: UploadFile = File(...),
    top_k: int = Form(20),
    state: AppState = Depends(get_app_state),
) -> SearchResponse:
    start_time = time.time()

    image_bytes = await sketch_image.read()
    img = Image.open(BytesIO(image_bytes))

    results = state.search_service.search(
        sketch_image=img,
        top_k=top_k,
    )

    query_time_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        results=[
            SearchResultItem(
                doc_id=r.doc_id,
                doc_name=r.doc_name,
                page_num=r.page_num,
                score=r.score,
                thumbnail_url=r.thumbnail_url,
            )
            for r in results
        ],
        query_time_ms=query_time_ms,
        total_indexed_pages=state.colqwen_index.page_count,
    )
