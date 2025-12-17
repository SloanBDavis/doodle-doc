from __future__ import annotations

import numpy as np


def extract_regions(
    img: np.ndarray,
    overlap_pct: float = 0.1,
) -> dict[str, np.ndarray]:
    """
    Extract full page + 4 quadrants with overlap.

    Quadrant layout:
    +-----+-----+
    | q1  | q2  |
    +-----+-----+
    | q3  | q4  |
    +-----+-----+
    """
    h, w = img.shape[:2]

    overlap_x = int(w * overlap_pct)
    overlap_y = int(h * overlap_pct)

    mid_x, mid_y = w // 2, h // 2

    regions = {
        "full": img,
        "q1": img[0 : mid_y + overlap_y, 0 : mid_x + overlap_x],
        "q2": img[0 : mid_y + overlap_y, mid_x - overlap_x : w],
        "q3": img[mid_y - overlap_y : h, 0 : mid_x + overlap_x],
        "q4": img[mid_y - overlap_y : h, mid_x - overlap_x : w],
    }

    return regions
