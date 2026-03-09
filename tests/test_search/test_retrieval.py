from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.core.models import Document
from doodle_doc.ingestion.colqwen_utils import compact_embedding, maxsim_score
from doodle_doc.search.retrieval import SearchService


def test_maxsim_score_ignores_zero_padded_rows() -> None:
    query_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    padded_doc_emb = np.array(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    compact_doc_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert compact_embedding(padded_doc_emb).shape == (2, 2)
    assert maxsim_score(query_emb, padded_doc_emb) == pytest.approx(
        maxsim_score(query_emb, compact_doc_emb),
    )


class _FakeEmbedder:
    def __init__(self, query_emb: np.ndarray) -> None:
        self.query_emb = query_emb

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def embed_single(self, img: Image.Image) -> np.ndarray:
        return self.query_emb


class _FakeIndex:
    def __init__(self, embeddings: dict[tuple[str, int], np.ndarray]) -> None:
        self.embeddings = embeddings
        self.page_count = len(embeddings)

    def all_page_keys(self) -> list[tuple[str, int]]:
        return list(self.embeddings)

    def get(self, doc_id: str, page_num: int) -> np.ndarray | None:
        return self.embeddings.get((doc_id, page_num))


class _FakeDb:
    def get_document(self, doc_id: str) -> Document:
        return Document(
            doc_id=doc_id,
            path=f"/tmp/{doc_id}.png",
            sha256="",
            modified_time=datetime.now(),
            num_pages=1,
        )


def test_search_service_ranks_with_zero_padded_embeddings(settings: Settings) -> None:
    service = SearchService(
        settings=settings,
        embedder=_FakeEmbedder(np.array([[1.0, 0.0]], dtype=np.float32)),
        index=_FakeIndex(
            {
                ("page_bad", 0): np.array(
                    [[0.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
                    dtype=np.float32,
                ),
                ("page_good", 0): np.array([[1.0, 0.0]], dtype=np.float32),
            }
        ),
        db=_FakeDb(),
    )

    results = service.search(Image.new("RGB", (8, 8), "white"), top_k=2)

    assert [result.doc_id for result in results] == ["page_good", "page_bad"]
    assert np.isfinite(results[0].score)
    assert np.isfinite(results[1].score)
