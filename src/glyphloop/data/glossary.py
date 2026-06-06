"""The vocabulary — one source of truth.

The original repo had two competing phrase lists (``unused/fixed_phrases.py`` vs a
newer ``glossary.py``) that disagreed, silently breaking label<->index alignment.
Here there is exactly one: the closed set of ~1,700 phrases in
``data/vocab/authentic_phrases.txt``. Everything else (symbols, words, n-grams and
their index maps) is derived from it, deterministically and cached.

The "phrase" is the unit the classifier predicts; the n-gram / word heads are the
auxiliary multi-label targets of the "hydra" model used at inference for
pattern-whitelisting (see :mod:`glyphloop.inference.postprocess`).
"""

from __future__ import annotations

from functools import lru_cache

from glyphloop.config import vocab_dir

PHRASES_FILE = vocab_dir() / "authentic_phrases.txt"


@lru_cache(maxsize=1)
def load_phrases() -> tuple[str, ...]:
    """Load the closed phrase vocabulary (sorted, de-duplicated)."""
    text = PHRASES_FILE.read_text(encoding="utf-8")
    phrases = {line.strip() for line in text.splitlines() if line.strip()}
    return tuple(sorted(phrases))


# Module-level convenience; identical to ``load_phrases()`` but as a list.
phrase_list: list[str] = list(load_phrases())


def strip_non_alnum(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c == " ")


@lru_cache(maxsize=1)
def get_symbol_set() -> frozenset[str]:
    return frozenset.union(*[frozenset(phrase) for phrase in load_phrases()])


@lru_cache(maxsize=1)
def get_word_set() -> frozenset[str]:
    return frozenset.union(*[frozenset(phrase.split()) for phrase in load_phrases()])


@lru_cache(maxsize=8)
def get_gram_set(n: int) -> frozenset[str]:
    """All character n-grams occurring within any word of the vocabulary.

    (Fixes the original ``util.py`` bug ``return list(*grams)`` — a crash for any
    ``len(grams) != 1`` — with a correct, deterministic frozenset.)
    """
    grams: set[str] = set()
    for w in get_word_set():
        grams.update(w[x : x + n] for x in range(0, len(w) - (n - 1)))
    return frozenset(grams)


def _indexed(items, *, null: bool) -> dict[str, int]:
    prefix = ["null"] if null else []
    return {tok: i for i, tok in enumerate(prefix + sorted(items))}


@lru_cache(maxsize=1)
def get_symbol_dict() -> dict[str, int]:
    return _indexed(get_symbol_set(), null=True)


@lru_cache(maxsize=1)
def get_word_dict() -> dict[str, int]:
    return _indexed(get_word_set(), null=True)


@lru_cache(maxsize=8)
def get_gram_dict(n: int) -> dict[str, int]:
    return _indexed(get_gram_set(n), null=False)


@lru_cache(maxsize=1)
def get_phrase_dict() -> dict[str, int]:
    return {phrase: i for i, phrase in enumerate(load_phrases())}


@lru_cache(maxsize=1)
def get_simplified_phrase_dict() -> dict[str, int]:
    return {strip_non_alnum(phrase): i for i, phrase in enumerate(load_phrases())}


def _reverse(d: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in d.items()}


def get_reverse_symbol_dict() -> dict[int, str]:
    return _reverse(get_symbol_dict())


def get_reverse_word_dict() -> dict[int, str]:
    return _reverse(get_word_dict())


def get_reverse_gram_dict(n: int) -> dict[int, str]:
    return _reverse(get_gram_dict(n))


def get_reverse_phrase_dict() -> dict[int, str]:
    return _reverse(get_phrase_dict())
