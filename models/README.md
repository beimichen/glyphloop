# Models

Trained weights are **not committed** to this repository, by design:

- They are large binary artifacts that don't belong in git history.
- This project is a *method* showcase; the headline numbers below are reported as
  results from the original 2022 runs, whose 1.3M-image dataset no longer exists
  (see the [Dataset Card](../docs/DATASET_CARD.md) and
  [Method writeup](../docs/METHOD.md)).

`.gitignore` keeps `*.h5` / `*.onnx` / `*.keras` out of this folder.

## What would live here

| File | Produced by | Used by |
|------|-------------|---------|
| `teacher.weights.h5` | `glyphloop selftrain` (final round) | distillation, evaluation |
| `student.weights.h5` | `glyphloop distill` | inference, ONNX export |
| `student.onnx` | `glyphloop export -w student.weights.h5` | TF-free inference |

## How to obtain weights

Train them from scratch (needs an unlabeled image pool and the `[train]` extra):

```bash
just train-pretrain                       # round 0: synthetic bootstrap
just selftrain data/raw/unlabeled         # rounds 1..N: confidence-gated self-training
just distill runs/selftrain/teacher.weights.h5   # teacher -> email-sized student
just export  runs/distill/student.weights.h5     # -> models/student.onnx
```

Or drop a compatible Keras student `.h5` in here and run
`glyphloop export -w models/student.weights.h5`.
