"""Color palettes and value<->color encodings used across the synthesis layers."""

import numpy as np
from numpy.random import normal, permutation

background_superimposable_characters = [chr(x) for x in range(32, 128) if chr(x).isalpha()]

background_color_discounts = [20, 40, 60]

background_color_templates = [
    (240, 240, 216),
    (240, 216, 240),
    (216, 240, 240),
    (240, 216, 216),
    (216, 240, 216),
    (216, 216, 240),
]

foreground_color_templates = [
    (176, 80, 80),
    (80, 176, 80),
    (80, 80, 176),
]


def random_color_from_template(template, sigma):
    def r(mu, sigma):
        return min(255, max(0, int(normal(mu, sigma))))

    return tuple(r(x, sigma) for x in permutation(template)[0])


def rgba_to_rgb(rgba):
    return np.array([rgba[0], rgba[1], rgba[2]], dtype=int)


def value_to_grayscale(v: int):
    # assuming a maximum of 64 symbol classes, spread across 0..252
    return 4 * v


def grayscale_to_value(v: int):
    return v // 4


def value_to_rgb(v: int):
    # assuming we have 16^3 options (4096)
    b = 16 * (v % 16)
    v //= 16
    g = 16 * (v % 16)
    v //= 16
    r = 16 * (v % 16)
    return r, g, b


def rgb_to_value(rgb: tuple):
    r, g, b = rgb
    return (round(r / 16) << 8) + (round(g / 16) << 4) + round(b / 16)
