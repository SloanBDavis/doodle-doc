from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from doodle_doc.ingestion.index import FAISSIndex


@pytest.mark.skip(reason="TODO FAISS crashing on my local - needs investigation")
def test_faiss_index_add_and_search():
    index = FAISSIndex(embedding_dim=128)

    embeddings = np.random.randn(10, 128).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    metadata = [{"doc_id": f"doc_{i}", "page_num": i} for i in range(10)]

    index.add(embeddings, metadata)

    assert index.size == 10

    results = index.search(embeddings[0], k=5)
    assert len(results) == 5
    assert results[0][0]["doc_id"] == "doc_0"
    assert results[0][1] > 0.99


@pytest.mark.skip(reason="TODO FAISS crashes on my local")
def test_faiss_index_save_load(temp_dir: Path):
    index = FAISSIndex(embedding_dim=64)

    embeddings = np.random.randn(5, 64).astype(np.float32)
    metadata = [{"id": i} for i in range(5)]
    index.add(embeddings, metadata)

    index.save(temp_dir)

    loaded = FAISSIndex.load(temp_dir)
    assert loaded.size == 5
    assert loaded.embedding_dim == 64

    results = loaded.search(embeddings[0], k=1)
    assert results[0][0]["id"] == 0
