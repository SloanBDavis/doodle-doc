from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FAISSIndex:
    """FAISS index for SigLIP2 embeddings."""

    def __init__(self, embedding_dim: int = 1152) -> None:
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.id_to_metadata: list[dict[str, Any]] = []

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add embeddings with metadata."""
        assert embeddings.shape[0] == len(metadata)
        assert embeddings.shape[1] == self.embedding_dim

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index.add(embeddings)
        self.id_to_metadata.extend(metadata)

    def search(
        self,
        query: np.ndarray,
        k: int = 100,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Search for similar embeddings.

        Returns: List of (metadata, score) tuples, sorted by score descending
        """
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)

        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.id_to_metadata[idx], float(score)))

        return results

    def save(self, path: Path) -> None:
        """Save index and metadata to disk."""
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))

        with open(path / "metadata.json", "w") as f:
            json.dump(self.id_to_metadata, f)

    @classmethod
    def load(cls, path: Path) -> FAISSIndex:
        """Load index and metadata from disk."""
        instance = cls()

        instance.index = faiss.read_index(str(path / "faiss.index"))
        instance.embedding_dim = instance.index.d

        with open(path / "metadata.json") as f:
            instance.id_to_metadata = json.load(f)

        return instance

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal
