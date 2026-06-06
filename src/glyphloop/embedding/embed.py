"""Turn images into vectors for clustering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def pixel_embedding(image_paths, size: tuple[int, int] = (32, 16)) -> np.ndarray:
    """A cheap, deterministic, TensorFlow-free embedding: downscaled flattened pixels.

    Good enough to demonstrate that same-phrase renders cluster together (and to
    build the committed t-SNE figure) without needing a trained encoder. Returns
    an ``(N, size[0]*size[1])`` float32 array in ``[0, 1]``.
    """
    vectors = []
    for p in image_paths:
        with Image.open(p) as im:
            im = im.convert("L").resize(size)
            vectors.append(np.asarray(im, dtype=np.float32).reshape(-1) / 255.0)
    return np.stack(vectors) if vectors else np.empty((0, size[0] * size[1]), dtype=np.float32)


def model_embedding(encoder, image_paths: list[Path], batch_size: int = 256) -> np.ndarray:
    """Embed images with a trained Keras ``encoder`` (the autoencoder bottleneck)."""
    from glyphloop.data.features import batch_features

    chunks = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        chunks.append(encoder.predict(batch_features(chunk), verbose=0))
    return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.float32)
