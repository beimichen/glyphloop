"""Evaluate a classifier on a held-out, human-verified split.

Reports top-1 and top-5 accuracy. By construction the evaluation set must be
human-verified and leakage-safe (see :mod:`glyphloop.data.splits`) — never
pseudo-labeled — so the numbers reflect generalization, not the model grading its
own homework.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from glyphloop.modeling.train import load_classification_dataset


def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argsort(probs, axis=1)[:, -k:]
    hits = np.any(topk == labels[:, None], axis=1)
    return float(hits.mean())


def evaluate(weights: Path, data_dir: Path, kind: str = "student") -> dict[str, float]:
    from glyphloop.modeling.architectures import build_student, build_teacher

    model = build_teacher() if kind == "teacher" else build_student()
    model.load_weights(weights)

    x, y = load_classification_dataset(data_dir)
    probs = model.predict(x, batch_size=256, verbose=0)
    metrics = {
        "n": float(len(y)),
        "top1": top_k_accuracy(probs, y, 1),
        "top5": top_k_accuracy(probs, y, 5),
    }
    print(f"n={int(metrics['n'])}  top-1={metrics['top1']:.4f}  top-5={metrics['top5']:.4f}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="glyphloop evaluate", description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="held-out dataset dir")
    ap.add_argument("--kind", choices=["teacher", "student"], default="student")
    args = ap.parse_args(argv)
    evaluate(args.weights, args.data, args.kind)


if __name__ == "__main__":
    main()
