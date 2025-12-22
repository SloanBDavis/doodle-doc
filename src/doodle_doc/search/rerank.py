from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image

from doodle_doc.core.models import SearchResult

if TYPE_CHECKING:
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

logger = logging.getLogger(__name__)


class ColQwen2Reranker:
    """ColQwen2 reranker for Stage 2.

    Uses late-interaction scoring to rerank Stage 1 results by computing
    pairwise similarity between query sketch and full-resolution page images.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0-hf",
        batch_size: int = 8,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._model: ColQwen2ForRetrieval | None = None
        self._processor: ColQwen2Processor | None = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model into memory."""
        if self._model is not None:
            return

        logger.info(f"Loading ColQwen2 model: {self.model_name} on {self.device}")

        from transformers import ColQwen2ForRetrieval, ColQwen2Processor

        dtype = torch.float16 if self.device == "mps" else torch.bfloat16

        self._model = ColQwen2ForRetrieval.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device,
        ).eval()

        self._processor = ColQwen2Processor.from_pretrained(self.model_name)

        logger.info("ColQwen2 model loaded")

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("ColQwen2 model unloaded")

    def _load_page_image(
        self, rendered_dir: Path, doc_id: str, page_num: int
    ) -> Image.Image | None:
        """Load rendered page image."""
        image_path = rendered_dir / doc_id / f"{page_num}.png"
        if not image_path.exists():
            logger.warning(f"Page image not found: {image_path}")
            return None
        return Image.open(image_path).convert("RGB")

    def rerank(
        self,
        results: list[SearchResult],
        sketch_image: Image.Image,
        rendered_dir: Path,
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Rerank results using ColQwen2 late-interaction scoring.

        Args:
            results: Stage 1 search results to rerank
            sketch_image: User's sketch as PIL Image
            rendered_dir: Path to rendered page images directory
            top_k: Number of results to return
        """
        if not results:
            return []

        if not self.is_loaded():
            self.load()

        assert self._model is not None
        assert self._processor is not None

        page_images: list[Image.Image] = []
        valid_results: list[SearchResult] = []

        for result in results:
            img = self._load_page_image(rendered_dir, result.doc_id, result.page_num)
            if img is not None:
                page_images.append(img)
                valid_results.append(result)

        if not valid_results:
            logger.warning("No valid page images found for reranking")
            return results[:top_k]

        try:
            with torch.no_grad():
                query_inputs = self._processor(images=[sketch_image]).to(self.device)
                query_emb = self._model(**query_inputs).embeddings

                all_scores: list[float] = []
                for i in range(0, len(page_images), self.batch_size):
                    batch = page_images[i : i + self.batch_size]
                    doc_inputs = self._processor(images=batch).to(self.device)
                    doc_emb = self._model(**doc_inputs).embeddings
                    scores = self._processor.score_retrieval(query_emb, doc_emb)
                    all_scores.extend(scores[0].tolist())

        except Exception as e:
            logger.error(f"Error computing rerank scores: {e}")
            return results[:top_k]

        scored = list(zip(valid_results, all_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked: list[SearchResult] = []
        for result, score in scored[:top_k]:
            reranked.append(
                SearchResult(
                    doc_id=result.doc_id,
                    doc_name=result.doc_name,
                    page_num=result.page_num,
                    score=float(score),
                    stage="reranked",
                    thumbnail_url=result.thumbnail_url,
                )
            )

        return reranked
