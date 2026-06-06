"""Active-learning selection logic — uncertainty + discovery/verification queues."""

from __future__ import annotations

import numpy as np

from glyphloop.active.discovery import discovery_queue
from glyphloop.active.mine import (
    margin,
    prediction_entropy,
    rank_by_uncertainty,
    summarize_clusters,
)
from glyphloop.active.verification import Propagation, verification_queue


def test_entropy_is_higher_for_uniform_rows():
    probs = np.array([[0.5, 0.5], [0.99, 0.01]])
    ent = prediction_entropy(probs)
    assert ent[0] > ent[1]


def test_margin_is_smaller_for_borderline_rows():
    probs = np.array([[0.5, 0.5], [0.99, 0.01]])
    m = margin(probs)
    assert m[0] < m[1]


def test_rank_by_uncertainty_entropy_then_margin_agree_on_most_unsure():
    probs = np.array([[0.99, 0.01], [0.6, 0.4], [0.5, 0.5]])
    assert rank_by_uncertainty(probs, "entropy")[0] == 2
    assert rank_by_uncertainty(probs, "margin")[0] == 2


def test_summarize_clusters_ignores_noise_and_sorts_by_size():
    labels = [0, 0, 0, 1, 1, -1, -1, -1]
    stats = summarize_clusters(labels, known_clusters={1})
    assert stats[0].cluster_id == 0 and stats[0].size == 3
    assert stats[0].has_known_label is False
    assert any(s.cluster_id == 1 and s.has_known_label for s in stats)


def test_discovery_queue_surfaces_only_large_unlabeled_clusters():
    labels = [0] * 12 + [1] * 12 + [2] * 3  # cluster 2 is too small
    q = discovery_queue(labels, known_clusters={1}, min_size=10)
    ids = [d.cluster_id for d in q]
    assert ids == [0]  # 1 is known, 2 is too small


def test_verification_queue_orders_least_similar_first():
    props = [
        Propagation("a", 0, "cat", 0.9),
        Propagation("b", 0, "cat", 0.4),
        Propagation("c", 0, "cat", 0.7),
    ]
    q = verification_queue(props)
    assert [v.item_id for v in q] == ["b", "c", "a"]
