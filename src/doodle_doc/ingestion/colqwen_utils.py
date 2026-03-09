from __future__ import annotations

import math

import numpy as np
import torch


def compact_embedding(embedding: np.ndarray) -> np.ndarray:
    """Drop padded zero rows from a ColQwen multi-vector embedding."""
    if embedding.ndim != 2 or embedding.shape[0] == 0:
        return embedding

    row_norms = np.linalg.norm(embedding, axis=1)
    keep_mask = row_norms > 0
    if keep_mask.all():
        return embedding

    return embedding[keep_mask]


def maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Compute a stable late-interaction score for two image embeddings."""
    query_compact = compact_embedding(query_emb)
    doc_compact = compact_embedding(doc_emb)

    if query_compact.size == 0 or doc_compact.size == 0:
        return float("-inf")

    query_tensor = torch.from_numpy(query_compact)
    doc_tensor = torch.from_numpy(doc_compact)

    query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=-1)
    doc_norm = torch.nn.functional.normalize(doc_tensor, p=2, dim=-1)
    sim_matrix = torch.matmul(query_norm, doc_norm.T)

    if torch.isnan(sim_matrix).any():
        return float("-inf")

    score = float(sim_matrix.max(dim=1).values.sum().item())
    if math.isnan(score):
        return float("-inf")

    return score
