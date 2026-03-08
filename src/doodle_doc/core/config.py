from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Rendering
    render_dpi: int = 150
    max_pages_per_doc: int = 500

    # Retrieval
    default_result_k: int = 20

    # ColQwen2
    colqwen_model: str = "vidore/colqwen2-v1.0-hf"
    colqwen_batch_size: int = 4

    # Synthetic generation
    synth_model: str = "gemini-2.5-flash-image"
    synth_prompt_version: str = "v2"

    # Evaluation
    eval_regression_threshold: float = 0.05

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

    @property
    def colqwen_index_dir(self) -> Path:
        return self.index_dir / "colqwen"


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
