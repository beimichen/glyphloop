"""Cluster embeddings (OPTICS) and project them to 2D (t-SNE) for inspection.

Needs the ``[viz]`` extra (scikit-learn). OPTICS is density-based: it discovers
how many clusters exist and marks outliers as ``-1`` — appropriate when you don't
know the number of phrases ahead of time (the cold-start premise).
"""

from __future__ import annotations

import numpy as np


def optics_labels(
    vectors: np.ndarray, *, min_samples: int = 5, max_eps: float = np.inf
) -> np.ndarray:
    """Density-based cluster assignment; ``-1`` marks noise/outliers."""
    from sklearn.cluster import OPTICS

    if len(vectors) == 0:
        return np.empty((0,), dtype=int)
    return OPTICS(min_samples=min_samples, max_eps=max_eps).fit(vectors).labels_


def tsne_2d(vectors: np.ndarray, *, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    """Project high-dimensional embeddings to 2D for plotting."""
    from sklearn.manifold import TSNE

    perplexity = min(perplexity, max(5.0, (len(vectors) - 1) / 3))
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed).fit_transform(vectors)
