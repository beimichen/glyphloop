"""Inference: turn model outputs into a ranked phrase guess.

For the multi-head "hydra" model, the auxiliary symbol/n-gram/word heads are not
decoration — they constrain the phrase prediction at inference time via
pattern-whitelisting and geometric-mean log-probability reranking
(:mod:`glyphloop.inference.postprocess`).
"""
