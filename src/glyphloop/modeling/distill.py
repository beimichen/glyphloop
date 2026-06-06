"""Knowledge distillation: squeeze the accurate teacher into an email-sized student.

This wasn't an abandoned experiment in the original project — it was load-bearing.
The self-trained teacher was accurate but heavy; distillation produced the
~9 MB student that was actually handed off. The student learns from the teacher's
*soft* probability distribution (temperature-scaled), which carries more signal
than hard labels alone.

TensorFlow is imported lazily; this module imports cheaply.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from glyphloop.config import paths
from glyphloop.modeling.train import load_classification_dataset


def teacher_soft_targets(
    teacher, x: np.ndarray, temperature: float, batch_size: int = 256
) -> np.ndarray:
    """Temperature-softened teacher probabilities over ``x``."""
    probs = teacher.predict(x, batch_size=batch_size, verbose=0)
    logits = np.log(np.clip(probs, 1e-9, 1.0))
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def distill(
    teacher_weights: Path,
    data_dir: Path,
    *,
    temperature: float = 4.0,
    epochs: int = 5,
    batch_size: int = 64,
    out: Path | None = None,
) -> Path:
    """Train a fresh student to match the teacher's soft targets. Returns weights path."""
    from tensorflow.keras.losses import KLDivergence
    from tensorflow.keras.optimizers import Adam

    from glyphloop.modeling.architectures import build_student, build_teacher

    teacher = build_teacher()
    teacher.load_weights(teacher_weights)
    teacher.trainable = False

    x, _ = load_classification_dataset(data_dir)
    soft = teacher_soft_targets(teacher, x, temperature)

    student = build_student()
    # Recompile the student to learn the soft distribution (vs its sparse-CE default).
    student.compile(optimizer=Adam(1e-3), loss=KLDivergence())
    student.fit(x, soft, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    out = out or (paths.runs / "distill" / "student.weights.h5")
    out.parent.mkdir(parents=True, exist_ok=True)
    student.save_weights(out)
    print(f"Distilled student -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="glyphloop distill", description=__doc__)
    ap.add_argument("--teacher", type=Path, required=True, help="teacher .weights.h5")
    ap.add_argument(
        "--data", type=Path, default=paths.synthetic, help="dataset dir of rendered PNGs"
    )
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    distill(
        args.teacher,
        args.data,
        temperature=args.temperature,
        epochs=args.epochs,
        out=args.out,
    )


if __name__ == "__main__":
    main()
