"""Turn CAPTCHA images into model-ready feature tensors.

The model input is a 128x64 single-channel image scaled to ``[0, 1]``. The
original code did this with a nested per-pixel Python loop; this is the same
transform, vectorized with NumPy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from glyphloop.config import IMAGE_HEIGHT, IMAGE_SIZE, IMAGE_WIDTH


def image_to_features(image: Image.Image) -> np.ndarray:
    """Grayscale, resize to 128x64 and scale to ``[0, 1]`` as ``(W, H, 1)`` float32."""
    image = image.convert("L").resize(IMAGE_SIZE)
    # PIL is (H, W); transpose so the array is (W, H) to match the model's (X, Y).
    arr = np.asarray(image, dtype=np.float32).T / 255.0
    return arr.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)


def load_features(path: str | Path) -> np.ndarray:
    """Load an image file and return its feature tensor."""
    with Image.open(path) as image:
        return image_to_features(image)


def batch_features(paths) -> np.ndarray:
    """Stack features for many image paths into a ``(N, W, H, 1)`` batch."""
    return (
        np.stack([load_features(p) for p in paths])
        if paths
        else np.empty((0, IMAGE_WIDTH, IMAGE_HEIGHT, 1), dtype=np.float32)
    )
