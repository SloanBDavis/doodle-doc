from __future__ import annotations

import random

from PIL import Image

from doodle_doc.eval.pseudo_queries import PseudoQueryConfig, PseudoQueryGenerator


class TestExtractRandomCrop:
    def test_crop_within_bounds(self) -> None:
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))

        config = PseudoQueryConfig(min_crop_ratio=0.2, max_crop_ratio=0.3)
        generator = PseudoQueryGenerator.__new__(PseudoQueryGenerator)
        generator.config = config

        rng = random.Random(42)
        crop, (x0, y0, x1, y1) = generator._extract_random_crop(img, rng)

        assert 0 <= x0 < x1 <= 400
        assert 0 <= y0 < y1 <= 400
        assert crop.size[0] == x1 - x0
        assert crop.size[1] == y1 - y0

    def test_crop_size_in_range(self) -> None:
        img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))

        config = PseudoQueryConfig(
            min_crop_ratio=0.15,
            max_crop_ratio=0.40,
            exclude_margins_pct=0.0,
        )
        generator = PseudoQueryGenerator.__new__(PseudoQueryGenerator)
        generator.config = config

        rng = random.Random(42)

        for _ in range(10):
            crop, _ = generator._extract_random_crop(img, rng)
            width, height = crop.size
            ratio = width / 1000
            assert 0.15 <= ratio <= 0.40

    def test_deterministic_with_seed(self) -> None:
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))

        config = PseudoQueryConfig(min_crop_ratio=0.2, max_crop_ratio=0.3)
        generator = PseudoQueryGenerator.__new__(PseudoQueryGenerator)
        generator.config = config

        rng1 = random.Random(42)
        _, box1 = generator._extract_random_crop(img, rng1)

        rng2 = random.Random(42)
        _, box2 = generator._extract_random_crop(img, rng2)

        assert box1 == box2


class TestPseudoQueryConfig:
    def test_default_values(self) -> None:
        config = PseudoQueryConfig()
        assert config.num_queries == 100
        assert config.min_crop_ratio == 0.15
        assert config.max_crop_ratio == 0.40
        assert config.seed == 42
        assert config.exclude_margins_pct == 0.05

    def test_custom_values(self) -> None:
        config = PseudoQueryConfig(
            num_queries=50,
            min_crop_ratio=0.1,
            max_crop_ratio=0.5,
            seed=123,
        )
        assert config.num_queries == 50
        assert config.seed == 123
