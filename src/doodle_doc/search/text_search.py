from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 index for text layer search."""

    def __init__(self) -> None:
        self.corpus: list[list[str]] = []
        self.metadata: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

    def add(self, text: str, metadata: dict[str, Any]) -> None:
        """Add a document to the index."""
        tokens = text.lower().split()
        self.corpus.append(tokens)
        self.metadata.append(metadata)

    def build(self) -> None:
        """Build the BM25 index after adding all documents."""
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, k: int = 100) -> list[tuple[dict[str, Any], float]]:
        """Search for documents matching query."""
        if not self.bm25:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        top_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.metadata[idx], float(scores[idx])))

        return results

    def save(self, path: Path) -> None:
        """Save index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump((self.corpus, self.metadata), f)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """Load index from disk."""
        instance = cls()
        pkl_path = path / "bm25.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                instance.corpus, instance.metadata = pickle.load(f)
            instance.build()
        return instance
