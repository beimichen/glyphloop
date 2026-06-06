"""Build a balanced synthetic CAPTCHA dataset from the phrase vocabulary.

This is the *data-engineering* half of the project: render each phrase enough
times, at randomized difficulty, that every class is represented — the bootstrap
corpus the round-0 teacher learns to read glyphs from.

Rendered images are written under ``data/processed/synthetic`` (gitignored — no
image corpora are committed). The phrase label is base64-encoded into the
filename so it round-trips losslessly regardless of punctuation.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from pathlib import Path

from numpy.random import randint
from tqdm import tqdm

from glyphloop.config import IMAGE_SIZE, paths
from glyphloop.data.glossary import load_phrases
from glyphloop.data.synthesis import render_text

MAX_DIFFICULTY = 5


def fs_safe(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c == " ").strip()


def encode_label(phrase: str) -> str:
    return base64.urlsafe_b64encode(phrase.encode("utf-8")).decode("ascii")


def decode_label(token: str) -> str:
    return base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")


def label_from_filename(name: str) -> str | None:
    """Recover the phrase from a ``... phrase=<b64> ...png`` filename, or None."""
    if 'phrase="' not in name:
        return None
    token = name.split('phrase="')[-1].split('"')[0]
    try:
        return decode_label(token)
    except Exception:
        return None


def existing_counts(out_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not out_dir.exists():
        return counts
    for f in out_dir.glob("*.png"):
        phrase = label_from_filename(f.name)
        if phrase is not None:
            counts[phrase] += 1
    return counts


def render_one(phrase: str, out_dir: Path) -> Path | None:
    """Render ``phrase`` once at a random difficulty and save it. None on failure."""
    difficulty = int(randint(MAX_DIFFICULTY))
    try:
        captcha, _, _, _, _ = render_text(IMAGE_SIZE, difficulty, phrase)
    except Exception:
        return None
    tag = int(randint(2**30))
    path = out_dir / f'title="{fs_safe(phrase)}", phrase="{encode_label(phrase)}" {tag}.png'
    captcha.save(path)
    return path


def build_synthetic_dataset(per_phrase: int = 100, out_dir: Path | None = None) -> int:
    """Render until every phrase has at least ``per_phrase`` examples. Returns the count made."""
    out_dir = out_dir or paths.synthetic
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = existing_counts(out_dir)
    phrases = load_phrases()
    todo = [(p, per_phrase - counts.get(p, 0)) for p in phrases]
    work = [p for p, deficit in todo for _ in range(max(0, deficit))]

    made = 0
    for phrase in tqdm(work, desc="rendering synthetic CAPTCHAs"):
        if render_one(phrase, out_dir) is not None:
            made += 1
    return made


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="glyphloop synth", description=__doc__)
    ap.add_argument("--per-phrase", type=int, default=100, help="target examples per phrase")
    ap.add_argument(
        "--out", type=Path, default=None, help="output dir (default: data/processed/synthetic)"
    )
    args = ap.parse_args(argv)

    made = build_synthetic_dataset(args.per_phrase, args.out)
    print(f"Rendered {made} synthetic CAPTCHAs into {args.out or paths.synthetic}")


if __name__ == "__main__":
    main()
