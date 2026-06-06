"""Composite the four synthesis layers into one CAPTCHA image.

``render_text`` is the package entry point. Fonts are discovered via
:func:`glyphloop.config.fonts_dir` (no more hardcoded ``./fonts``).
"""

from PIL import Image

from glyphloop.config import fonts_dir
from glyphloop.data.synthesis.background import generate_background
from glyphloop.data.synthesis.foreground import generate_foreground
from glyphloop.data.synthesis.superground import generate_superground
from glyphloop.data.synthesis.textground import generate_textground


def available_fonts() -> list[str]:
    directory = fonts_dir()
    fonts = [str(p) for p in directory.glob("*.ttf")]
    if not fonts:
        raise FileNotFoundError(f"No .ttf fonts found in {directory}")
    return fonts


def flatten_layers(img_dim, background, textground, foreground, superground):
    """Stack the layers per pixel: superground > foreground (where text is) > background."""
    composite = Image.new(mode="RGB", size=img_dim)
    for x in range(img_dim[0]):
        for y in range(img_dim[1]):
            superpixel = superground.getpixel((x, y))
            textpixel = textground.getpixel((x, y))
            forepixel = foreground.getpixel((x, y))

            if superpixel != (0, 0, 0, 0):  # superground present: ignore text color here
                if superpixel[-1] == 0:  # transparent outline -> show background
                    composite.putpixel((x, y), value=background.getpixel((x, y)))
                else:
                    composite.putpixel((x, y), value=tuple(superpixel[:-1]))
            elif textpixel != 0:  # text present -> paint it the foreground color
                if forepixel[-1] == 0:  # transparent foreground -> show background
                    composite.putpixel((x, y), value=background.getpixel((x, y)))
                else:
                    composite.putpixel((x, y), value=tuple(forepixel[:-1]))
            else:
                composite.putpixel((x, y), value=background.getpixel((x, y)))
    return composite


def render_text(img_dim, distortion_level, desired_phrase):
    """Render ``desired_phrase`` as a synthetic CAPTCHA.

    Returns ``(captcha, phrase, mask, labels, textground)`` where ``captcha`` is
    the composited RGB image and ``labels`` / ``mask`` are the aligned per-pixel
    symbol-label and word-mask layers.
    """
    fonts = available_fonts()

    background = generate_background(img_dim, fonts, distortion_level)
    foreground = generate_foreground(img_dim, distortion_level)
    superground = generate_superground(img_dim, distortion_level)
    phrase, mask, labels, textground = generate_textground(
        img_dim, fonts, distortion_level, desired_phrase
    )

    captcha = flatten_layers(img_dim, background, textground, foreground, superground)
    return captcha, phrase, mask, labels, textground
