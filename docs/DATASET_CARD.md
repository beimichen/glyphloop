# Dataset Card — glyphloop

## Overview

The project uses two kinds of data over a **closed vocabulary** of ~1,700
pop-culture phrases (`src/glyphloop/data/vocab/authentic_phrases.txt`, the only
data committed to this repo):

| Subset | Source | Labels | Committed? |
|--------|--------|--------|------------|
| **Synthetic** | The in-repo generator renders each phrase with layered noise, occluders and mesh distortions | Exact (generation-time) | No — rendered on demand |
| **Real (historical)** | A large pool of unlabeled SolveMedia-style distorted-phrase images | Machine pseudo-labels (train) / human-verified (test) | No — gone, not redistributable |

## The vocabulary

`authentic_phrases.txt` — 1,727 unique English pop-culture phrases (≈2,100 unique
words). It is the single source of truth for the glossary (symbols, n-grams, words,
phrases and their index maps in `glyphloop.data.glossary`). The original repo had
two competing phrase lists that silently disagreed; this consolidates to one.

## How the real data was labeled (historical)

1. **Bootstrap** a reader on synthetic data.
2. **Pseudo-label** the unlabeled pool; keep only top-confidence predictions
   (`glyphloop.selftrain`) — the empirical precision of that slice is what made
   self-training work.
3. **Human-verify** a held-out subset for honest evaluation. In the unbuilt
   cold-start variant, humans answer only two questions — *discovery* ("what does
   this cluster say?", grows train) and *verification* ("was this propagation
   right?", grows test) — surfaced by `glyphloop.active`.

## Splits

`glyphloop.data.splits.split_by_group` partitions by a stable **group key** so all
variants of one source stay on the same side of the train/test boundary, and
`assert_test_is_human_verified` refuses any machine-pseudo-labeled item in the test
split. Together these keep evaluation from being inflated by near-duplicate leakage
or by the model grading its own pseudo-labels.

> A caution learned the hard way (see [METHOD.md](METHOD.md)): because the source
> served a *finite* pool of pre-rendered templates, even a clean group-split
> "held-out" set consisted of near-duplicates of training items — so the reported
> generalization gap was ~0 for reasons of the data source, not the split logic.

## Availability & licensing

- **No image corpora are committed.** Synthetic images are generated locally
  (`just sample`) and gitignored; the real historical pool no longer exists.
- The committed phrase list contains only short, public pop-culture phrases.

## Known biases

English-only, closed-vocabulary, single historical rendering style. Nothing about
this dataset generalizes to other text domains — by design; it is a controlled
testbed for the *method*.
