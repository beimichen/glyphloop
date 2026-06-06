"""Confidence-gated self-training — the core method.

Synthetic bootstrap -> pseudo-label a large unlabeled pool -> keep only the
high-confidence predictions (the precision gate) -> second-stage train on that
real distribution -> iterate. High precision at the top of the confidence
distribution is the whole game: it's what makes pseudo-labeling close the
synthetic->real gap instead of amplifying noise.
"""

from glyphloop.selftrain.pseudolabel import PseudoLabel, select_confident

__all__ = ["PseudoLabel", "select_confident"]
