from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Rendering
    render_dpi: int = 150
    max_pages_per_doc: int = 500

    # Preprocessing
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # Embedding
    siglip_model: str = "google/siglip-so400m-patch14-384"
    siglip_batch_size: int = 32
    embedding_dim: int = 1152

    # Retrieval
    stage1_top_k: int = 100
    default_result_k: int = 20

    # Reranking
    colqwen_model: str = "vidore/colqwen2-v1"
    colqwen_lazy_load: bool = True
    rerank_batch_size: int = 8

    # Text search
    enable_text_boost: bool = True
    text_boost_weight: float = 0.3

    # Index
    faiss_index_type: Literal["IndexFlatIP"] = "IndexFlatIP"

    # Paths
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    config_path: Path | None = None

    model_config = {"env_prefix": "DOODLE_DOC_"}

    @property
    def rendered_dir(self) -> Path:
        return self.data_dir / "rendered"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"


def load_settings_from_yaml(path: Path) -> Settings:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Settings(**data, config_path=path)


@lru_cache
def get_settings() -> Settings:
    default_config = Path("configs/default.yaml")
    if default_config.exists():
        return load_settings_from_yaml(default_config)
    return Settings()
