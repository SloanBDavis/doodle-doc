from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from doodle_doc.core.config import Settings
from doodle_doc.core.database import Database
from doodle_doc.core.models import SearchResult
from doodle_doc.ingestion.colqwen_embed import ColQwen2Embedder
from doodle_doc.ingestion.colqwen_index import ColQwen2Index


def _prepare_query_image(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


class SearchService:
    """ColQwen2 retrieval service."""

    def __init__(
        self,
        settings: Settings,
        embedder: ColQwen2Embedder | None = None,
        index: ColQwen2Index | None = None,
        db: Database | None = None,
    ) -> None:
        self.settings = settings
        self._embedder = embedder
        self._index = index
        self._db = db

    @property
    def embedder(self) -> ColQwen2Embedder:
        if self._embedder is None:
            self._embedder = ColQwen2Embedder(model_name=self.settings.colqwen_model)
        return self._embedder

    @property
    def index(self) -> ColQwen2Index:
        if self._index is None:
            self._index = ColQwen2Index.load(self.settings.colqwen_index_dir)
        return self._index

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = Database(self.settings.index_dir / "metadata.sqlite")
        return self._db

    def search(
        self,
        sketch_image: Image.Image,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        top_k = top_k or self.settings.default_result_k
        if self.index.page_count == 0:
            return []

        if not self.embedder.is_loaded():
            self.embedder.load()

        query_emb = self.embedder.embed_single(_prepare_query_image(sketch_image))
        query_tensor = torch.from_numpy(query_emb).unsqueeze(0)

        scores: list[tuple[str, int, float]] = []
        for doc_id, page_num in self.index.all_page_keys():
            doc_emb = self.index.get(doc_id, page_num)
            if doc_emb is None:
                continue

            doc_tensor = torch.from_numpy(doc_emb).unsqueeze(0)
            score = self._compute_maxsim(query_tensor, doc_tensor)
            scores.append((doc_id, page_num, score))

        scores.sort(key=lambda item: item[2], reverse=True)

        results: list[SearchResult] = []
        for doc_id, page_num, score in scores[:top_k]:
            doc = self.db.get_document(doc_id)
            if doc is None:
                continue
            results.append(SearchResult(
                doc_id=doc_id,
                doc_name=Path(doc.path).name,
                page_num=page_num,
                score=float(score),
                thumbnail_url=f"/v1/thumb/{doc_id}/{page_num}",
            ))

        return results

    def _compute_maxsim(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor,
    ) -> float:
        query_norm = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
        doc_norm = torch.nn.functional.normalize(doc_emb, p=2, dim=-1)

        sim_matrix = torch.matmul(query_norm[0], doc_norm[0].T)
        max_sims = sim_matrix.max(dim=1).values
        return float(max_sims.sum().item())
