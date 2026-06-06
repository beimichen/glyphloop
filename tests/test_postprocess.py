"""The inference reranking math — pure, no model required."""

from __future__ import annotations

import numpy as np
import pytest

from glyphloop.inference import postprocess as pp


def test_phrase_patterns_extracts_grams_and_words():
    monos, bis, tris, words = pp.phrase_patterns("ab cd")
    assert monos == {"a", "b", "c", "d"}
    assert bis == {"ab", "cd"}
    assert tris == set()  # no word is 3+ chars
    assert words == {"ab", "cd"}


def test_geometric_mean_is_between_the_probs():
    dist = np.array([0.25, 0.81])
    index = {"a": 0, "b": 1}
    gm = pp.geometric_mean_prob(dist, index, {"a", "b"})
    assert gm == pytest.approx(np.sqrt(0.25 * 0.81))


def test_geometric_mean_unknown_token_is_zero():
    dist = np.array([0.5])
    assert pp.geometric_mean_prob(dist, {"a": 0}, {"z"}) == 0.0


def test_geometric_mean_empty_tokenset_is_zero():
    assert pp.geometric_mean_prob(np.array([0.5]), {"a": 0}, set()) == 0.0


def test_build_pattern_phrase_map_links_tokens_to_phrases():
    mapping = pp.build_pattern_phrase_map(["ab", "ac"])
    assert mapping["a"] == {"ab", "ac"}
    assert mapping["ab"] == {"ab"}


def test_rerank_prefers_phrase_supported_by_all_heads():
    phrases = ["cat", "dog"]
    # phrase head is ambivalent; the n-gram/word heads strongly support "cat".
    phrase_probs = np.array([0.5, 0.5])
    head_dicts = {
        "mono": {"c": 0, "a": 1, "t": 2, "d": 3, "o": 4, "g": 5},
        "bi": {"ca": 0, "at": 1, "do": 2, "og": 3},
        "tri": {"cat": 0, "dog": 1},
        "words": {"cat": 0, "dog": 1},
    }
    head_dists = {
        "mono": np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1]),
        "bi": np.array([0.9, 0.9, 0.1, 0.1]),
        "tri": np.array([0.9, 0.1]),
        "words": np.array([0.9, 0.1]),
    }
    ranked = pp.rerank_phrases(phrases, phrase_probs, head_dists, head_dicts)
    assert ranked[0][1] == "cat"
    assert sum(score for score, _ in ranked) == pytest.approx(1.0)
