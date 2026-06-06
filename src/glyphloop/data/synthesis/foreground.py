"""Foreground layer: a solid color overlaid with occluding polygons and lines."""

from numpy.random import choice, randint, random
from PIL import Image, ImageDraw

from glyphloop.data.synthesis.colors import foreground_color_templates, random_color_from_template
from glyphloop.data.synthesis.helpers import get_polygon
from glyphloop.data.synthesis.rng import random_center, random_radius, random_rgb, random_rgba


def generate_foreground(img_dim, distortion_level, fg_shapes=None, fg_lines=None):
    if fg_shapes is None:
        fg_shapes = distortion_level
    if fg_lines is None:
        fg_lines = distortion_level

    fg_color = (*random_color_from_template(foreground_color_templates, 8), 255)
    foreground = Image.new(mode="RGBA", size=img_dim, color=fg_color)

    for _ in range(fg_shapes):
        ImageDraw.Draw(foreground).polygon(
            get_polygon(random_center(img_dim), random_radius(img_dim), choice([3, 4, 5, 6])),
            fill=random_rgba(),
        )
    for _ in range(fg_lines):
        ImageDraw.Draw(foreground).line(
            [random_center(img_dim) for _ in range(2)],
            fill=(*random_rgb(), 0 if random() < 0.2 else 255),
            width=randint(3, 7),
        )
    return foreground
