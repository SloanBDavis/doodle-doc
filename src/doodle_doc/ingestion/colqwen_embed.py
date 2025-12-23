from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

logger = logging.getLogger(__name__)


class ColQwen2Embedder:
    """ColQwen2 embedder for document indexing.

    Produces multi-vector embeddings (one vector per image patch) for
    late-interaction scoring. Unlike SigLIP2 which produces single vectors.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0-hf",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name

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
        if self._model is not None:
            return

        logger.info(f"Loading ColQwen2 embedder: {self.model_name} on {self.device}")

        from transformers import ColQwen2ForRetrieval, ColQwen2Processor

        dtype = torch.float16 if self.device == "mps" else torch.bfloat16

        self._model = ColQwen2ForRetrieval.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device,
        ).eval()

        self._processor = ColQwen2Processor.from_pretrained(self.model_name)
        logger.info("ColQwen2 embedder loaded")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("ColQwen2 embedder unloaded")

    @torch.no_grad()
    def embed_single(self, img: Image.Image) -> np.ndarray:
        """Embed a single image.

        Returns:
            numpy array of shape (num_patches, hidden_dim)
        """
        if not self.is_loaded():
            self.load()

        assert self._model is not None
        assert self._processor is not None

        inputs = self._processor(images=[img.convert("RGB")]).to(self.device)
        outputs = self._model(**inputs)

        embeddings = outputs.embeddings[0].cpu().numpy()
        return embeddings

    @torch.no_grad()
    def embed_batch(
        self,
        images: list[Image.Image],
        batch_size: int = 4,
    ) -> list[np.ndarray]:
        """Embed a batch of images.

        Returns:
            List of numpy arrays, each (num_patches, hidden_dim)
        """
        if not self.is_loaded():
            self.load()

        assert self._model is not None
        assert self._processor is not None

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = [img.convert("RGB") for img in images[i : i + batch_size]]
            inputs = self._processor(images=batch).to(self.device)
            outputs = self._model(**inputs)

            for j in range(len(batch)):
                emb = outputs.embeddings[j].cpu().numpy()
                all_embeddings.append(emb)

        return all_embeddings

    @property
    def processor(self) -> ColQwen2Processor:
        if not self.is_loaded():
            self.load()
        assert self._processor is not None
        return self._processor
