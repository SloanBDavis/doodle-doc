from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from doodle_doc.api.deps import get_app_state
from doodle_doc.api.main import app
from doodle_doc.core.models import SearchResult


class DummyIndex:
    def __init__(self, page_count: int) -> None:
        self.page_count = page_count


class FakeSearchService:
    def search(self, sketch_image: Image.Image, top_k: int | None = None) -> list[SearchResult]:
        assert sketch_image.size == (16, 16)
        assert top_k == 5
        return [
            SearchResult(
                doc_id="doc-1",
                doc_name="notes.pdf",
                page_num=2,
                score=0.87,
                thumbnail_url="/v1/thumb/doc-1/2",
            )
        ]


def test_search_endpoint_returns_accurate_only_schema() -> None:
    state = get_app_state()
    original_service = state._search_service
    original_index = state._colqwen_index

    state._search_service = FakeSearchService()  # type: ignore[assignment]
    state._colqwen_index = DummyIndex(page_count=11)  # type: ignore[assignment]

    image = Image.new("RGBA", (16, 16), (255, 255, 255, 0))
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    client = TestClient(app)
    response = client.post(
        "/v1/search",
        files={"sketch_image": ("sketch.png", buf.getvalue(), "image/png")},
        data={"top_k": "5"},
    )

    state._search_service = original_service
    state._colqwen_index = original_index

    assert response.status_code == 200
    data = response.json()
    assert data["total_indexed_pages"] == 11
    assert data["results"][0]["doc_id"] == "doc-1"
    assert "stage" not in data["results"][0]
