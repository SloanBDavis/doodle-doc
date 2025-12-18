from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across all lists
    Higher k = less aggressive boosting of top results
    """
    scores: dict[str, float] = defaultdict(float)

    for result_list in result_lists:
        for rank, (key, _) in enumerate(result_list, start=1):
            scores[key] += 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results
