"""Pseudo-labeling and the confidence gate.

The empirical observation that made the whole project work: at the very top of
the confidence distribution, predictions were almost never wrong. So we keep only
predictions above a high threshold and feed *those* back as training labels. The
gate is a precision knob, not an accuracy one — we happily throw away most of the
pool to keep what's left clean.

The pure selection logic here (:func:`select_confident`) needs only NumPy, so it
is unit-tested without TensorFlow. Running a model over images lives in
:func:`pseudo_label_pool`, which imports TF lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PseudoLabel:
    index: int  # row index into the pool
    label: int  # predicted phrase index
    confidence: float


def select_confident(probs: np.ndarray, threshold: float = 0.99) -> list[PseudoLabel]:
    """Keep only argmax predictions whose top probability is ``>= threshold``.

    ``probs`` is ``(N, num_classes)`` softmax output. Returns one
    :class:`PseudoLabel` per retained row, sorted by descending confidence.
    """
    if probs.ndim != 2:
        raise ValueError(f"expected (N, C) probabilities, got shape {probs.shape}")
    labels = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    keep = np.nonzero(conf >= threshold)[0]
    picks = [PseudoLabel(int(i), int(labels[i]), float(conf[i])) for i in keep]
    picks.sort(key=lambda p: p.confidence, reverse=True)
    return picks


def gate_yield(probs: np.ndarray, threshold: float) -> float:
    """Fraction of the pool that clears the gate — how much signal a round adds."""
    if len(probs) == 0:
        return 0.0
    return float((probs.max(axis=1) >= threshold).mean())


def pseudo_label_pool(
    model, image_paths: list[Path], threshold: float = 0.99, batch_size: int = 256
):
    """Run ``model`` over images and return the confident pseudo-labels."""
    from glyphloop.data.features import batch_features

    picks: list[PseudoLabel] = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        probs = model.predict(batch_features(chunk), verbose=0)
        for pl in select_confident(probs, threshold):
            picks.append(PseudoLabel(start + pl.index, pl.label, pl.confidence))
    return picks
