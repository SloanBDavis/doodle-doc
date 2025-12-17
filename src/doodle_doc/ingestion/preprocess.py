from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def resize_with_padding(
    img: np.ndarray,
    target_size: tuple[int, int],
    pad_value: int = 255,
) -> np.ndarray:
    """Resize image maintaining aspect ratio, pad to square."""
    h, w = img.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    output = np.full((target_h, target_w), pad_value, dtype=np.uint8)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    output[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return output


def normalize_ink(
    img: Image.Image,
    clip_limit: float = 2.0,
    grid_size: int = 8,
    target_size: tuple[int, int] = (384, 384),
) -> np.ndarray:
    """
    Normalize handwritten page for embedding.

    Returns: numpy array (H, W, 3) in RGB format, uint8
    """
    img_np = np.array(img)

    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(gray)

    if np.mean(enhanced) < 127:
        enhanced = 255 - enhanced

    resized = resize_with_padding(enhanced, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    return rgb


def normalize_sketch(
    img: Image.Image,
    clip_limit: float = 2.0,
    grid_size: int = 8,
    target_size: tuple[int, int] = (384, 384),
) -> np.ndarray:
    """
    Normalize user sketch for query.
    Composites transparency to white, then applies same normalization.
    """
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return normalize_ink(img, clip_limit, grid_size, target_size)
