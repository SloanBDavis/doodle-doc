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
    text_query: str | None = Form(None),
    top_k: int = Form(20),
    use_rerank: bool = Form(False),
    state: AppState = Depends(get_app_state),
) -> SearchResponse:
    start_time = time.time()

    image_bytes = await sketch_image.read()
    img = Image.open(BytesIO(image_bytes))

    results = state.search_service.search(
        sketch_image=img,
        text_query=text_query,
        top_k=top_k,
        use_rerank=use_rerank,
    )

    query_time_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        results=[
            SearchResultItem(
                doc_id=r.doc_id,
                doc_name=r.doc_name,
                page_num=r.page_num,
                score=r.score,
                stage=r.stage,
                thumbnail_url=r.thumbnail_url,
            )
            for r in results
        ],
        query_time_ms=query_time_ms,
        total_indexed_pages=state.index.size // 5,
    )
