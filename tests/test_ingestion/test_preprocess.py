from __future__ import annotations

import numpy as np
from PIL import Image

from doodle_doc.ingestion.preprocess import normalize_ink, resize_with_padding


def test_resize_with_padding_square():
    img = np.zeros((100, 100), dtype=np.uint8)
    result = resize_with_padding(img, (200, 200))
    assert result.shape == (200, 200)


def test_resize_with_padding_landscape():
    img = np.zeros((100, 200), dtype=np.uint8)
    result = resize_with_padding(img, (100, 100))
    assert result.shape == (100, 100)


def test_resize_with_padding_portrait():
    img = np.zeros((200, 100), dtype=np.uint8)
    result = resize_with_padding(img, (100, 100))
    assert result.shape == (100, 100)


def test_normalize_ink_returns_rgb(sample_image: Image.Image):
    result = normalize_ink(sample_image, target_size=(384, 384))
    assert result.shape == (384, 384, 3)
    assert result.dtype == np.uint8


def test_normalize_ink_white_background(sample_image: Image.Image):
    result = normalize_ink(sample_image)
    # Corners should be mostly white (high values)
    corners = [result[0, 0], result[0, -1], result[-1, 0], result[-1, -1]]
    for corner in corners:
        assert corner.mean() > 200
