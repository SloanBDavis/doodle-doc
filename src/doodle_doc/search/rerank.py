from __future__ import annotations

from doodle_doc.core.models import SearchResult

# TODO: Implement ColQwen2 reranker in Milestone 3


class ColQwen2Reranker:
    """ColQwen2 reranker for Stage 2 (M3 milestone)."""

    def __init__(self, model_name: str = "vidore/colqwen2-v1") -> None:
        self.model_name = model_name
        self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model into memory."""
        # TODO: Implement model loading
        pass

    def unload(self) -> None:
        """Unload the model from memory."""
        self._model = None

    def rerank(
        self,
        results: list[SearchResult],
        sketch_path: str,
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Rerank results using ColQwen2."""
        # TODO: Implement reranking
        return results[:top_k]
