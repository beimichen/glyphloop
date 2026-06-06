"""Synthetic CAPTCHA generator — the bootstrap data engine.

Composites four layers into a distorted-phrase image that mimics the old
SolveMedia style: a noisy ``background``, the distorted ``textground`` (the
phrase itself), an occluding ``foreground`` (shapes + lines), and a top-layer
``superground`` (outlines). Mesh ``deformations`` (lean / squash / perspective)
are applied per character, per word and per phrase.

Entry point: :func:`glyphloop.data.synthesis.text_rendering.render_text`.
"""

from glyphloop.data.synthesis.text_rendering import render_text

__all__ = ["render_text"]
