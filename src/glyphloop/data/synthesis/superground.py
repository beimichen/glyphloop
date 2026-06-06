"""Top layer: transparent outline shapes (ellipses, rectangles, polygons)."""

# We shuffle a list of bound PIL draw methods with numpy.permutation; numpy's stubs
# don't model that, so its call/argument typing is suppressed for this module.
# pyright: reportCallIssue=false, reportArgumentType=false

from numpy.random import permutation, random
from PIL import Image, ImageDraw

from glyphloop.data.synthesis.rng import random_center, random_rgb


def _ordered_box(img_dim):
    """Two random corners normalized to ``[(x0, y0), (x1, y1)]`` with x0<=x1, y0<=y1.

    PIL's ``ellipse``/``rectangle`` require the box ordered; the original code
    passed raw random corners and leaned on a broad ``try/except`` to skip the
    ones that came out reversed.
    """
    (xa, ya), (xb, yb) = random_center(img_dim), random_center(img_dim)
    return [(min(xa, xb), min(ya, yb)), (max(xa, xb), max(ya, yb))]


def generate_superground(img_dim, distortion_level, sg_shapes=None):
    if sg_shapes is None:
        sg_shapes = distortion_level

    superground = Image.new(mode="RGBA", size=img_dim, color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(superground)
    shapes = [draw.ellipse, draw.rectangle, draw.polygon]

    for _ in range(sg_shapes):
        permutation(shapes)[0](
            _ordered_box(img_dim),
            outline=(*random_rgb(), 0 if random() < 0.2 else 255),
        )

    return superground
