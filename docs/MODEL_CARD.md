# Model Card — glyphloop distorted-text reader

## Model summary

A classifier that reads a distorted-text image and predicts which phrase, from a
**closed vocabulary** of ~1,700 phrases, it shows. It is the product of a
semi-supervised pipeline — synthetic pretraining, confidence-gated self-training
on unlabeled real images, then knowledge distillation into a small deployable
student.

- **Task:** image → phrase classification over a fixed vocabulary (~1,700 classes)
- **Input:** 128×64 single-channel image, scaled to `[0, 1]`
- **Architectures** (`src/glyphloop/modeling/architectures.py`):
  - *Teacher* — ResNet-style CNN (accurate, heavy)
  - *Student* — depthwise/DenseNet-style CNN (~9 MB; the deployed model)
  - *Hydra* — shared backbone with symbol / n-gram / word / phrase heads, whose
    auxiliary outputs drive inference-time pattern-whitelisting + reranking
  - *Autoencoder* — encoder used as the cold-start image embedder
- **Frameworks:** TensorFlow/Keras (training), ONNX Runtime (inference)
- **License:** MIT

## Intended use

- **In scope:** recognizing distorted text drawn from a *known, finite* vocabulary
  — a constrained-OCR / closed-set classification problem, and a teaching example
  of synthetic→real self-training and distillation.
- **Out of scope:** open-vocabulary OCR, handwriting, natural-scene text, and any
  use against a live service. SolveMedia is defunct; this models a fixed historical
  image domain only.

## Training data

- **Synthetic:** images rendered by the in-repo generator (`glyphloop.data.synthesis`)
  — distorted phrases over layered noise with occluders — used for the round-0
  bootstrap. Ground truth is exact (generation-time labels).
- **Real (historical):** a large pool of unlabeled SolveMedia-style images,
  **pseudo-labeled** by the model under a high-confidence gate. This pool is gone
  and was never redistributable; no images are committed here.
- Pseudo-labels only ever entered **train**; evaluation used a human-verified,
  leakage-safe held-out split (`glyphloop.data.splits`).

## Training procedure

1. **Round 0 — synthetic pretrain.** Learn glyph appearance on the synthetic
   distribution (`configs/train/pretrain_synthetic.yaml`).
2. **Rounds 1..N — confidence-gated self-training.** Pseudo-label the real pool,
   keep predictions with top probability ≥ 0.99, fine-tune on that slice
   (`configs/selftrain/round.yaml`). Two rounds drove essentially all the gain.
3. **Distillation.** Train the small student to match the teacher's
   temperature-softened distribution (`glyphloop.modeling.distill`).

## Evaluation results

From the original 2022 runs (reported as historical; see the README and
[METHOD.md](METHOD.md) — **not re-trainable in this repo**):

| Metric | Value |
|--------|-------|
| Accuracy on ~1,000 held-out real images | ~99.1% |
| Distilled student size | ~9 MB |

> Important context: the very high accuracy turned out to reflect a property of the
> *data source* — the images were served from a finite pool of pre-rendered
> templates, so a large scrape enumerated the population rather than sampling a
> distribution. Recognition, not generalization, was doing the work. This is
> analyzed in [METHOD.md](METHOD.md#the-punchline-a-security-economics-failure-not-an-ml-one)
> and is a deliberate part of the writeup's honesty.

## Limitations & ethical considerations

- **Closed-vocabulary only.** The model classifies into a fixed phrase set; it does
  not read arbitrary text.
- **Domain-specific.** Trained on one historical image style; it will not transfer
  to other CAPTCHA designs or general OCR.
- **Historical metrics.** Numbers above are from the original runs and are not
  reproduced here (dataset/weights gone).
- **No misuse surface shipped.** The repo contains no automation or network code
  for interacting with any service.
