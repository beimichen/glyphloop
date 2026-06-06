"""Uncertainty mining and cluster bookkeeping — pure logic, unit-tested.

Two complementary signals decide what to label next:

- **Per-image uncertainty** (entropy / margin) ranks individual images so the
  most informative ones are reviewed first.
- **Cluster state** decides the *kind* of question: a dense cluster with no known
  label nearby is a discovery candidate; a cluster that inherited a propagated
  label is a verification candidate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


def prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of each row of a probability matrix. Higher = less sure."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def margin(probs: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 probability per row. Smaller = more borderline."""
    part = np.sort(probs, axis=1)
    return part[:, -1] - part[:, -2]


def rank_by_uncertainty(
    probs: np.ndarray, method: str = "entropy", top_k: int | None = None
) -> list[int]:
    """Return row indices ordered most-uncertain first."""
    if method == "entropy":
        scores = prediction_entropy(probs)  # larger = more uncertain
        order = np.argsort(scores)[::-1]
    elif method == "margin":
        scores = margin(probs)  # smaller = more uncertain
        order = np.argsort(scores)
    else:
        raise ValueError(f"unknown method: {method!r}")
    order = order.tolist()
    return order[:top_k] if top_k is not None else order


@dataclass(frozen=True)
class ClusterStats:
    cluster_id: int
    size: int
    has_known_label: bool


def summarize_clusters(labels, known_clusters: set[int]) -> list[ClusterStats]:
    """Tally cluster sizes and whether each already has a human-known label.

    ``labels`` is the per-item cluster assignment (``-1`` = OPTICS noise, ignored).
    """
    counts = Counter(int(c) for c in labels if int(c) != -1)
    return [
        ClusterStats(cid, size, cid in known_clusters)
        for cid, size in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
