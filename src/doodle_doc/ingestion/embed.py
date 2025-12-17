from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SigLIP2Embedder:
    """SigLIP2 image embedder for document search."""

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: str | None = None,
    ) -> None:
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_name = model_name

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.embedding_dim = self.model.config.vision_config.hidden_size

    @torch.no_grad()
    def embed_images(
        self,
        images: list[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed a batch of images.

        Args:
            images: List of numpy arrays (H, W, 3), uint8, RGB
            batch_size: Processing batch size

        Returns:
            numpy array of shape (N, embedding_dim), float32
        """
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            pil_images = [Image.fromarray(img) for img in batch]

            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.vision_model(**inputs)

            embeddings = outputs.pooler_output

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_single(self, img: np.ndarray) -> np.ndarray:
        """Embed a single image. Returns shape (embedding_dim,)."""
        result = self.embed_images([img], batch_size=1)
        return result[0]
