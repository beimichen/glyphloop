"""The self-training round orchestrator.

Round 0 bootstraps on synthetic data; each subsequent round pseudo-labels the
unlabeled real pool with the current model, keeps the confident slice, and
fine-tunes on it. Pseudo-labels only ever touch *train*; the human-verified
held-out set is the anchor we measure against every round, which is exactly what
makes clean accuracy-flattening (rather than train/test divergence) a signal that
confirmation bias never took hold.

Needs the ``[train]`` extra and an unlabeled image pool; imports TF lazily.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from glyphloop.config import paths, set_seed
from glyphloop.data.generate import encode_label, fs_safe
from glyphloop.data.glossary import get_reverse_phrase_dict
from glyphloop.selftrain.pseudolabel import pseudo_label_pool


@dataclass
class RoundResult:
    round: int
    pool_size: int
    confident: int
    yield_frac: float


def stage_pseudo_labeled(picks, pool_paths, reverse_phrases, staged_dir: Path) -> int:
    """Copy confident pool images into a training dir, labelled like synthetic data."""
    staged_dir.mkdir(parents=True, exist_ok=True)
    for pl in picks:
        phrase = reverse_phrases[pl.label]
        src = pool_paths[pl.index]
        dst = (
            staged_dir
            / f'title="{fs_safe(phrase)}", phrase="{encode_label(phrase)}" {pl.index}.png'
        )
        shutil.copyfile(src, dst)
    return len(picks)


def run_rounds(
    pool_dir: Path,
    teacher_weights: Path,
    *,
    rounds: int = 2,
    confidence: float = 0.99,
    epochs: int = 3,
    out_dir: Path | None = None,
) -> list[RoundResult]:
    from glyphloop.modeling.architectures import build_teacher
    from glyphloop.modeling.train import load_classification_dataset

    set_seed()
    out_dir = out_dir or (paths.runs / "selftrain")
    out_dir.mkdir(parents=True, exist_ok=True)
    reverse_phrases = get_reverse_phrase_dict()

    pool_paths = sorted(Path(pool_dir).glob("*.png"))
    staged = out_dir / "staged_train"

    model = build_teacher()
    model.load_weights(teacher_weights)

    results: list[RoundResult] = []
    for r in range(1, rounds + 1):
        picks = pseudo_label_pool(model, pool_paths, threshold=confidence)
        n = stage_pseudo_labeled(picks, pool_paths, reverse_phrases, staged)
        results.append(RoundResult(r, len(pool_paths), n, n / max(1, len(pool_paths))))
        print(
            f"[round {r}] pool={len(pool_paths)} confident={n} ({n / max(1, len(pool_paths)):.3%})"
        )

        x, y = load_classification_dataset(staged)
        model.fit(x, y, epochs=epochs, batch_size=64, validation_split=0.1)
        model.save_weights(out_dir / "teacher.weights.h5")

    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="glyphloop selftrain", description=__doc__)
    ap.add_argument("--pool", type=Path, required=True, help="dir of unlabeled CAPTCHA images")
    ap.add_argument("--teacher", type=Path, required=True, help="round-0 teacher .weights.h5")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--confidence", type=float, default=0.99)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    run_rounds(
        args.pool,
        args.teacher,
        rounds=args.rounds,
        confidence=args.confidence,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
