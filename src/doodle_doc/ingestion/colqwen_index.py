from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ColQwen2Index:
    """Storage for ColQwen2 multi-vector page embeddings.

    Storage layout:
    colqwen/
        manifest.json
        embeddings/
            {doc_id}_{page_num}.npy
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.embeddings_dir = index_dir / "embeddings"
        self.manifest_path = index_dir / "manifest.json"

        self._manifest: dict[str, Any] = {
            "version": 1,
            "model": "",
            "pages": {},
        }

    def _page_key(self, doc_id: str, page_num: int) -> str:
        return f"{doc_id}:{page_num}"

    def _embedding_filename(self, doc_id: str, page_num: int) -> str:
        return f"{doc_id}_{page_num}.npy"

    def add(
        self,
        doc_id: str,
        page_num: int,
        embedding: np.ndarray,
    ) -> None:
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        filename = self._embedding_filename(doc_id, page_num)
        filepath = self.embeddings_dir / filename

        np.save(filepath, embedding)

        key = self._page_key(doc_id, page_num)
        self._manifest["pages"][key] = {
            "file": filename,
            "shape": list(embedding.shape),
        }

    def get(self, doc_id: str, page_num: int) -> np.ndarray | None:
        key = self._page_key(doc_id, page_num)
        if key not in self._manifest["pages"]:
            return None

        filename = self._manifest["pages"][key]["file"]
        filepath = self.embeddings_dir / filename

        if not filepath.exists():
            return None

        return np.load(filepath)

    def get_batch(
        self,
        page_keys: list[tuple[str, int]],
    ) -> list[np.ndarray | None]:
        return [self.get(doc_id, page_num) for doc_id, page_num in page_keys]

    def has_page(self, doc_id: str, page_num: int) -> bool:
        key = self._page_key(doc_id, page_num)
        return key in self._manifest["pages"]

    def remove_by_doc_id(self, doc_id: str) -> int:
        removed = 0
        keys_to_remove = []

        for key, meta in self._manifest["pages"].items():
            if key.startswith(f"{doc_id}:"):
                filepath = self.embeddings_dir / meta["file"]
                if filepath.exists():
                    filepath.unlink()
                keys_to_remove.append(key)
                removed += 1

        for key in keys_to_remove:
            del self._manifest["pages"][key]

        return removed

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    @classmethod
    def load(cls, index_dir: Path) -> ColQwen2Index:
        instance = cls(index_dir)

        if instance.manifest_path.exists():
            with open(instance.manifest_path) as f:
                instance._manifest = json.load(f)

        return instance

    @property
    def page_count(self) -> int:
        return len(self._manifest["pages"])

    def set_model(self, model_name: str) -> None:
        self._manifest["model"] = model_name

    def all_page_keys(self) -> list[tuple[str, int]]:
        """Return all (doc_id, page_num) tuples in the index."""
        result = []
        for key in self._manifest["pages"]:
            doc_id, page_num_str = key.split(":")
            result.append((doc_id, int(page_num_str)))
        return result
