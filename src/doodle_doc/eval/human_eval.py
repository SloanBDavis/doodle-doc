from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

from doodle_doc.core.config import Settings
from doodle_doc.core.models import SearchResult
from doodle_doc.eval.metrics import RetrievalMetrics


RelevanceLabel = Literal["relevant", "partially_relevant", "not_relevant"]


@dataclass
class ResultAnnotation:
    doc_id: str
    page_num: int
    relevance: RelevanceLabel


@dataclass
class HumanAnnotation:
    query_id: str
    results: list[ResultAnnotation] = field(default_factory=list)
    notes: str = ""
    annotator: str = ""


@dataclass
class HumanEvalDataset:
    version: int = 1
    annotations: list[HumanAnnotation] = field(default_factory=list)


class HumanEvalManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def human_queries_dir(self) -> Path:
        return self.settings.data_dir / "eval" / "human_queries"

    @property
    def annotations_path(self) -> Path:
        return self.human_queries_dir / "annotations.json"

    def create_annotation_template(
        self,
        query_ids: list[str],
        search_results: dict[str, list[SearchResult]],
    ) -> Path:
        """Create annotation JSON template for human labeling."""
        self.human_queries_dir.mkdir(parents=True, exist_ok=True)

        annotations = []
        for query_id in query_ids:
            results = search_results.get(query_id, [])
            result_annotations = [
                {
                    "doc_id": r.doc_id,
                    "page_num": r.page_num,
                    "relevance": "not_relevant",
                }
                for r in results[:10]
            ]
            annotations.append({
                "query_id": query_id,
                "results": result_annotations,
                "notes": "",
                "annotator": "",
            })

        data = {
            "version": 1,
            "annotations": annotations,
        }

        with open(self.annotations_path, "w") as f:
            json.dump(data, f, indent=2)

        return self.annotations_path

    def load_annotations(self) -> HumanEvalDataset:
        """Load human annotations from JSON file."""
        if not self.annotations_path.exists():
            return HumanEvalDataset()

        with open(self.annotations_path) as f:
            data = json.load(f)

        annotations = []
        for ann in data.get("annotations", []):
            results = [
                ResultAnnotation(**r) for r in ann.get("results", [])
            ]
            annotations.append(HumanAnnotation(
                query_id=ann["query_id"],
                results=results,
                notes=ann.get("notes", ""),
                annotator=ann.get("annotator", ""),
            ))

        return HumanEvalDataset(
            version=data.get("version", 1),
            annotations=annotations,
        )

    def compute_metrics(
        self,
        dataset: HumanEvalDataset,
        relevance_threshold: RelevanceLabel = "relevant",
    ) -> RetrievalMetrics:
        """Compute metrics from human annotations.

        Args:
            dataset: Human evaluation dataset with annotations
            relevance_threshold: Minimum relevance to count as a hit.
                "relevant" = only exact matches
                "partially_relevant" = partial and exact matches
        """
        if not dataset.annotations:
            return RetrievalMetrics()

        threshold_values = {
            "relevant": ["relevant"],
            "partially_relevant": ["relevant", "partially_relevant"],
            "not_relevant": ["relevant", "partially_relevant", "not_relevant"],
        }
        valid_labels = threshold_values[relevance_threshold]

        recalls: dict[int, list[float]] = {1: [], 5: [], 10: [], 20: []}
        mrrs: list[float] = []

        for ann in dataset.annotations:
            for k in [1, 5, 10, 20]:
                hit = 0.0
                for result in ann.results[:k]:
                    if result.relevance in valid_labels:
                        hit = 1.0
                        break
                recalls[k].append(hit)

            mrr = 0.0
            for i, result in enumerate(ann.results):
                if result.relevance in valid_labels:
                    mrr = 1.0 / (i + 1)
                    break
            mrrs.append(mrr)

        num_queries = len(dataset.annotations)
        return RetrievalMetrics(
            recall_at_1=sum(recalls[1]) / num_queries if num_queries else 0.0,
            recall_at_5=sum(recalls[5]) / num_queries if num_queries else 0.0,
            recall_at_10=sum(recalls[10]) / num_queries if num_queries else 0.0,
            recall_at_20=sum(recalls[20]) / num_queries if num_queries else 0.0,
            mrr=sum(mrrs) / num_queries if num_queries else 0.0,
            num_queries=num_queries,
        )

    def save_dataset(self, dataset: HumanEvalDataset) -> None:
        """Save dataset back to disk."""
        self.human_queries_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "version": dataset.version,
            "annotations": [
                {
                    "query_id": ann.query_id,
                    "results": [asdict(r) for r in ann.results],
                    "notes": ann.notes,
                    "annotator": ann.annotator,
                }
                for ann in dataset.annotations
            ],
        }

        with open(self.annotations_path, "w") as f:
            json.dump(data, f, indent=2)
