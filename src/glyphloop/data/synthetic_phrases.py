"""Random glyph-string generator for the cold-start seed.

Chicken-and-egg problem: with zero known phrases you can't synthesize phrases —
but you *do* know the alphabet. So synthesize random lowercase strings, pretrain a
reader on them, and use that as the seed embedder that bootstraps clustering and
human-in-the-loop discovery (see :mod:`glyphloop.active`). This is what the
original ``mutable/synthetic_phrases`` file held.
"""

from __future__ import annotations

from numpy.random import choice, normal


def legal_synthetic_symbols() -> list[str]:
    """The alphabet we synthesize from — lowercase a-z (easy to widen later)."""
    return [chr(x) for x in range(ord("a"), ord("z") + 1)]


def _accumulate(
    substrings, num_sequences, length_mu, length_sigma, min_len, delimiter
) -> list[str]:
    out: set[str] = set()
    while len(out) < num_sequences:
        length = max(min_len, int(normal(length_mu, length_sigma)))
        out.add(delimiter.join(choice(substrings) for _ in range(length)))
    return sorted(out)


def synthetic_word_generator(num_words=1000, length_mu=5, length_sigma=2, min_len=1) -> list[str]:
    """Generate random 'words' from the legal symbol set."""
    return _accumulate(legal_synthetic_symbols(), num_words, length_mu, length_sigma, min_len, "")


def synthetic_phrase_generator(
    num_phrases=1000, length_mu=3, length_sigma=1, min_len=1
) -> list[str]:
    """Generate random multi-word 'phrases' from random 'words'."""
    words = synthetic_word_generator()
    return _accumulate(words, num_phrases, length_mu, length_sigma, min_len, " ")
