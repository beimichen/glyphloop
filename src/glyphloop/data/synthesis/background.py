"""Background layer: pastel noise with scattered, rotated glyphs."""

from numpy.random import normal, permutation, randint
from PIL import Image, ImageDraw, ImageFont

from glyphloop.data.synthesis.colors import (
    background_color_discounts,
    background_color_templates,
    background_superimposable_characters,
    random_color_from_template,
)


def generate_background(img_dim, bg_fonts, distortion_level, num_chars=None):
    """Build a noisy background.

    Strategy: draw on a canvas 5x the target size, stamp a character somewhere
    near the middle, rotate the whole canvas a random amount, repeat, then crop
    the center back out — so glyphs end up at every orientation.
    """
    if num_chars is None:
        num_chars = distortion_level

    def get_bgchar_color(bg_color):
        def r(mu, sigma):
            return int(normal(mu, sigma))

        discount = permutation(background_color_discounts)[0]
        return tuple(r(x - discount, 8) for x in bg_color)

    bg_color = random_color_from_template(background_color_templates, 4)
    bg_size = (img_dim[0] * 5, img_dim[1] * 5)
    background = Image.new(mode="RGB", size=bg_size, color=bg_color)

    for _ in range(num_chars):
        center = (
            normal(bg_size[0] // 2, bg_size[0] // 16),
            normal(bg_size[1] // 2, bg_size[1] // 16),
        )
        symbol = permutation(background_superimposable_characters)[0]
        ImageDraw.Draw(background).text(
            xy=center,
            text=symbol,
            fill=get_bgchar_color(bg_color),
            font=ImageFont.truetype(permutation(bg_fonts)[0], randint(50, 150)),
            align="center",
        )
        background = background.rotate(randint(0, 360), resample=Image.Resampling.BILINEAR)

    return background.crop(
        (
            bg_size[0] // 2 - img_dim[0] // 2,
            bg_size[1] // 2 - img_dim[1] // 2,
            bg_size[0] // 2 + img_dim[0] // 2,
            bg_size[1] // 2 + img_dim[1] // 2,
        )
    )
