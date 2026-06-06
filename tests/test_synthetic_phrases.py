"""Cold-start seed vocabulary generator."""

from __future__ import annotations

from glyphloop.data.synthetic_phrases import (
    legal_synthetic_symbols,
    synthetic_phrase_generator,
    synthetic_word_generator,
)


def test_legal_symbols_are_lowercase_alpha():
    syms = legal_synthetic_symbols()
    assert syms == [chr(c) for c in range(ord("a"), ord("z") + 1)]


def test_word_generator_makes_distinct_nonempty_words():
    words = synthetic_word_generator(num_words=50)
    assert len(words) == 50
    assert all(w and w.isalpha() for w in words)


def test_phrase_generator_makes_space_delimited_phrases():
    phrases = synthetic_phrase_generator(num_phrases=20)
    assert len(phrases) == 20
    assert all(all(part.isalpha() for part in p.split()) for p in phrases)
