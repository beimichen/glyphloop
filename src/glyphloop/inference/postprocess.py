"""Pure post-processing: combine the hydra heads into a ranked phrase guess.

The phrase head alone is a softmax over ~1,700 classes. The auxiliary heads
(which monograms / bigrams / trigrams / words are present) carry independent
evidence. We fuse them by, for each candidate phrase, taking the **geometric mean
of the head probabilities** for the tokens that phrase contains, then multiplying
those geometric means by the phrase-head probability. The geometric mean (an
arithmetic mean in log-space) rewards a phrase only when *all* of its constituent
tokens are supported — one missing trigram tanks the score — which is exactly the
behaviour you want for evidence fusion.

Everything here is pure NumPy/Python and unit-tested without a model.
"""

from __future__ import annotations

from math import log

import numpy as np


def phrase_patterns(phrase: str) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return the (monograms, bigrams, trigrams, words) present in ``phrase``."""
    words = phrase.split()
    monos = {w[i] for w in words for i in range(len(w))}
    bis = {w[i : i + 2] for w in words for i in range(len(w) - 1)}
    tris = {w[i : i + 3] for w in words for i in range(len(w) - 2)}
    return monos, bis, tris, set(words)


def geometric_mean_prob(dist: np.ndarray, index_of: dict[str, int], tokens: set[str]) -> float:
    """Geometric mean of ``dist`` over the given tokens; 0 if any token is unknown.

    Computed in log-space for stability: ``exp(mean(log p_t))``. Returns 0.0 for an
    empty token set (a phrase contributing no evidence on this head).
    """
    if not tokens:
        return 0.0
    total = 0.0
    for t in tokens:
        if t not in index_of:
            return 0.0
        p = float(dist[index_of[t]])
        total += log(p) if p > 0 else log(1e-12)
    return float(np.exp(total / len(tokens)))


def build_pattern_phrase_map(phrases) -> dict[str, set[str]]:
    """Map each token (mono/bi/tri/word) to the set of phrases that contain it."""
    mapping: dict[str, set[str]] = {}
    for phrase in phrases:
        for group in phrase_patterns(phrase):
            for token in group:
                mapping.setdefault(token, set()).add(phrase)
    return mapping


def rerank_phrases(
    phrases,
    phrase_probs: np.ndarray,
    head_dists: dict[str, np.ndarray],
    head_dicts: dict[str, dict[str, int]],
) -> list[tuple[float, str]]:
    """Fuse the phrase head with the n-gram/word heads into a ranked list.

    ``head_dists`` / ``head_dicts`` are keyed by ``"mono" | "bi" | "tri" | "words"``.
    Returns ``[(score, phrase), ...]`` sorted by descending fused score (normalized
    to sum to 1).
    """
    phrase_index = {p: i for i, p in enumerate(phrases)}
    scores = np.zeros(len(phrases), dtype=float)
    for phrase in phrases:
        monos, bis, tris, words = phrase_patterns(phrase)
        evidence = (
            geometric_mean_prob(head_dists["mono"], head_dicts["mono"], monos)
            * geometric_mean_prob(head_dists["bi"], head_dicts["bi"], bis)
            * geometric_mean_prob(head_dists["tri"], head_dicts["tri"], tris)
            * geometric_mean_prob(head_dists["words"], head_dicts["words"], words)
        )
        scores[phrase_index[phrase]] = float(phrase_probs[phrase_index[phrase]]) * evidence

    total = scores.sum()
    if total > 0:
        scores = scores / total
    ranked = sorted(((float(scores[i]), p) for i, p in enumerate(phrases)), reverse=True)
    return ranked


def top_k(ranked: list[tuple[float, str]], k: int = 5) -> list[tuple[float, str]]:
    return ranked[:k]
