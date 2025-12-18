from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from doodle_doc.core.config import Settings


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def settings(temp_dir: Path) -> Settings:
    return Settings(data_dir=temp_dir)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a simple test image with some dark strokes on white."""
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    pixels = img.load()
    # Draw some dark lines
    for x in range(100, 300):
        pixels[x, 200] = (0, 0, 0)
        pixels[200, x] = (0, 0, 0)
    return img


@pytest.fixture
def sample_numpy_image(sample_image: Image.Image) -> np.ndarray:
    return np.array(sample_image)
