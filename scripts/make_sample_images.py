"""Render a handful of synthetic CAPTCHAs to eyeball the generator.

Output goes to ``data/processed/synthetic`` which is **gitignored** — this repo
ships no image corpora. Use it to sanity-check that synthesis works on a fresh
clone (`just sample`).

Run:  python scripts/make_sample_images.py --n 8
"""

from __future__ import annotations

import argparse
import random

from glyphloop.config import paths, set_seed
from glyphloop.data.generate import render_one
from glyphloop.data.glossary import load_phrases


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir = paths.synthetic
    out_dir.mkdir(parents=True, exist_ok=True)

    phrases = list(load_phrases())
    made = 0
    for _ in range(args.n):
        if render_one(rng.choice(phrases), out_dir) is not None:
            made += 1
    print(f"Rendered {made}/{args.n} synthetic CAPTCHAs into {out_dir} (gitignored)")


if __name__ == "__main__":
    main()
