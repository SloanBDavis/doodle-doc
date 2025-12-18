from __future__ import annotations

import numpy as np

from doodle_doc.ingestion.regions import extract_regions


def test_extract_regions_returns_five():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    regions = extract_regions(img)
    assert len(regions) == 5
    assert set(regions.keys()) == {"full", "q1", "q2", "q3", "q4"}


def test_extract_regions_full_is_original():
    img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    regions = extract_regions(img)
    assert np.array_equal(regions["full"], img)


def test_extract_regions_quadrants_overlap():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    regions = extract_regions(img, overlap_pct=0.1)

    # With 10% overlap on 400px, quadrants should be larger than 200px
    for name in ["q1", "q2", "q3", "q4"]:
        h, w = regions[name].shape[:2]
        assert h > 200
        assert w > 200
