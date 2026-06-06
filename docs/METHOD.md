# Method — the part worth keeping

The dataset was a curiosity; the **method** is the point. This is a clean
semi-supervised recipe for going from *zero labels* to a small, accurate,
deployable reader — and an honest account of why it worked, including the twist
that the "ML win" was partly a property of the data source.

## The setup

A closed-vocabulary distorted-text problem: ~1,700 known phrases, rendered with
layered background noise, occluding shapes/lines and per-glyph mesh distortions.
The constraint matters — this is **classification over a known vocabulary**, not
open-ended OCR — and the whole method leans on it.

## 1. Synthetic bootstrap

We hand-built a synthetic generator (`glyphloop.data.synthesis`) that fakes the
image style: a noisy background, the distorted phrase (`textground`), occluding
shapes (`foreground`), and top-layer outlines (`superground`), composited with
mesh deformations (lean / squash / perspective). Pretraining on it gives a model
that can *read glyphs* on the wrong distribution — good enough to be a
**pseudo-labeler**, not yet a solver.

The biggest risk for synthetic-only approaches — "does the synthesis distribution
match reality?" — is **sidestepped**, because synthetic is only the bootstrap. The
real distribution enters through pseudo-labels.

## 2. Confidence-gated self-training

Pseudo-label a large pool of *unlabeled real* images with the current model, then
keep only the predictions above a high confidence threshold and train a second
stage on them (`glyphloop.selftrain`).

The empirical observation that makes this work: **at the top of the confidence
distribution, predictions were almost never wrong** (~1,000 sampled, near-zero
error). So the gate is a *precision* knob — we happily discard most of the pool to
keep a clean, large, real-distribution training set. That's what closes the
synthetic→real domain gap instead of amplifying noise.

```python
from glyphloop.selftrain import select_confident
keep = select_confident(probs, threshold=0.99)   # only the almost-never-wrong slice
```

## 3. Distillation

The self-trained teacher was accurate but heavy. Distillation
(`glyphloop.modeling.distill`) trains a small student to match the teacher's
temperature-softened distribution, producing the ~9 MB ("email-attachment-sized")
model that was actually handed off. This wasn't an abandoned experiment in the
original project — it was **load-bearing**.

## Did confirmation bias creep in? (it didn't)

Three rounds total: synthetic pretrain + two pseudo-label rounds. Big gain on
round 1, smaller-but-real gain on round 2, then it **flattened**.

The flattening *is* the evidence. Confirmation bias doesn't look like flattening —
it looks like train/self-consistency still improving while held-out accuracy
stalls or regresses (the model getting confidently wrong on a systematic slice).
Clean flattening on a **real, human-verified held-out set** means the
high-confidence slice stayed actually-correct round over round. The separation that
kept it safe: pseudo-labels only ever touched **train**; the held-out captchas were
the anchor (`glyphloop.data.splits.assert_test_is_human_verified` encodes that
rule).

## The cold-start sequel (embed → cluster → ask a human)

"What if you had a million images and *no phrase list at all*?" Use the model as an
embedder, cluster the images, and ask the human only the two questions machines are
worst at:

- **Discovery** — "here's a dense cluster with no label nearby; what does it say?"
  One answer mints a new vocabulary entry → feeds synthetic generation and
  pretraining. This grows **train**. (`glyphloop.active.discovery`)
- **Verification** — "I propagated label X across this cluster; was I right?"
  Confirmation promotes those into the held-out **test** set, keeping evaluation
  honest as the model drifts. (`glyphloop.active.verification`)

The two queries are orthogonal and both productive, and the loop is self-closing:
new label → synth data → better embedder → cleaner clusters → cheaper next label.
The hard part is cluster purity vs. the propagation assumption — and the
chicken-and-egg that the embedder is weakest *exactly* at cold start, before any
labels exist. (That's why the seed embedder is pretrained on random glyph-strings:
`glyphloop.data.synthetic_phrases`.)

## The punchline: a security-economics failure, not an ML one

Looking at the final model's clustering, the embedding spontaneously **subdivided a
single phrase by the font used**. That's the tell.

Truly on-the-fly generation would smear font/distortion/placement into a continuum
— one fuzzy blob per phrase. Crisp font sub-clusters mean a **finite set of
pre-rendered templates served repeatedly**. The cluster geometry reverse-engineered
the backend: the images were almost certainly pre-computed and cached to save
compute, not generated on the fly.

With a finite asset pool and a large scrape, you don't sample a distribution — you
**enumerate a population** (coupon-collector: you see essentially every template
many times). So the "held-out" set wasn't truly held out; it was near-duplicates of
training items. The generalization gap collapses to ~0 because there's no unseen
manifold to generalize to. A confident prediction wasn't *inference*, it was
*recognition* — "I've seen this exact asset before." Memorization is normally the
failure mode; here it was optimal, because the world really was a lookup table. ~99%
in a tiny model stops being surprising: you don't need much capacity to index a
finite set.

**The principle:** the security of such a system rests entirely on its asset space
being effectively infinite. The moment it pre-computes to save compute, it turns a
generation problem into a database — and a database is enumerable.

And the method was robust either way: had the source generated on the fly, the
synthetic-bootstrap → confidence-gated self-training → distillation pipeline was
already a legitimately good reader. The data source just happened to hand over the
easy version of the problem.

## Takeaways

- **Synthetic data is a bootstrap, not the destination.** Let the real
  distribution in through confidence-gated pseudo-labels.
- **A precision gate beats a bigger model** when the top of the confidence
  distribution is clean.
- **Keep the test set human-verified and leakage-safe**, always — it's the only
  thing that tells you whether self-training is learning or self-deceiving.
- **Read your clusters.** The embedding's *structure* told us more about the system
  than the accuracy number did.
