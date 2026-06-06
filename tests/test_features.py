"""Image -> feature tensor transform."""

from __future__ import annotations

import numpy as np
from PIL import Image

from glyphloop.config import IMAGE_HEIGHT, IMAGE_WIDTH
from glyphloop.data.features import batch_features, image_to_features


def test_image_to_features_shape_and_range():
    img = Image.new("RGB", (300, 120), (128, 200, 60))
    feats = image_to_features(img)
    assert feats.shape == (IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    assert feats.dtype == np.float32
    assert feats.min() >= 0.0 and feats.max() <= 1.0


def test_batch_features_empty_is_well_shaped():
    out = batch_features([])
    assert out.shape == (0, IMAGE_WIDTH, IMAGE_HEIGHT, 1)
