"""End-to-end smoke test of the synthesis engine (the crown jewel).

Renders one low-distortion CAPTCHA and checks the geometry of the outputs. Kept
fast by using distortion level 0 (no occluders / mesh distortions).
"""

from __future__ import annotations

import numpy as np

from glyphloop.config import IMAGE_SIZE, set_seed
from glyphloop.data.synthesis import render_text
from glyphloop.data.synthesis.text_rendering import available_fonts


def test_fonts_are_bundled():
    assert len(available_fonts()) > 0


def test_render_text_produces_aligned_layers_for_a_known_phrase():
    set_seed(0)
    captcha, phrase, mask, labels, textground = render_text(IMAGE_SIZE, 0, "abracadabra")

    assert phrase == "abracadabra"
    assert captcha.size == IMAGE_SIZE
    assert labels.size == IMAGE_SIZE
    assert mask.size == IMAGE_SIZE

    # the rendered text layer must actually contain some glyph pixels
    assert np.asarray(textground).max() > 0
