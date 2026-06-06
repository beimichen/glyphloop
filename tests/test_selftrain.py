"""The confidence gate — the precision knob at the heart of self-training."""

from __future__ import annotations

import numpy as np

from glyphloop.selftrain.pseudolabel import gate_yield, select_confident


def test_select_confident_keeps_only_above_threshold():
    probs = np.array(
        [
            [0.995, 0.005],  # confident -> kept
            [0.60, 0.40],  # unsure -> dropped
            [0.005, 0.995],  # confident -> kept (class 1)
        ]
    )
    picks = select_confident(probs, threshold=0.99)
    assert [p.index for p in picks] == [0, 2]
    assert [p.label for p in picks] == [0, 1]


def test_select_confident_sorted_by_descending_confidence():
    probs = np.array([[0.991, 0.009], [0.999, 0.001]])
    picks = select_confident(probs, threshold=0.99)
    assert [p.index for p in picks] == [1, 0]
    assert picks[0].confidence >= picks[1].confidence


def test_gate_yield_is_the_kept_fraction():
    probs = np.array([[0.999, 0.001], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    assert gate_yield(probs, 0.99) == 0.25


def test_gate_yield_empty_pool():
    assert gate_yield(np.empty((0, 3)), 0.99) == 0.0
