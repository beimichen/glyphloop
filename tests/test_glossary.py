"""The vocabulary is the single source of truth — guard its invariants."""

from __future__ import annotations

from glyphloop.data import glossary


def test_phrase_list_is_nonempty_sorted_and_unique():
    phrases = glossary.load_phrases()
    assert len(phrases) > 1000  # ~1,700 known phrases
    assert list(phrases) == sorted(phrases)
    assert len(set(phrases)) == len(phrases)


def test_phrase_dict_is_a_bijection():
    d = glossary.get_phrase_dict()
    assert len(d) == len(glossary.load_phrases())
    assert sorted(d.values()) == list(range(len(d)))
    rev = glossary.get_reverse_phrase_dict()
    assert all(rev[i] == p for p, i in d.items())


def test_get_gram_set_handles_many_grams():
    # The original util.py had `return list(*grams)`, which crashes for any
    # gram-set whose size != 1. This must just work.
    for n in (1, 2, 3):
        grams = glossary.get_gram_set(n)
        assert len(grams) > 1
        assert all(len(g) <= n for g in grams)


def test_gram_dicts_are_contiguous_indexes():
    for n in (1, 2, 3):
        d = glossary.get_gram_dict(n)
        assert sorted(d.values()) == list(range(len(d)))


def test_symbol_and_word_dicts_have_null_at_zero():
    assert glossary.get_symbol_dict()["null"] == 0
    assert glossary.get_word_dict()["null"] == 0
