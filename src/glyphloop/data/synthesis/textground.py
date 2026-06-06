"""Textground layer: render a phrase as distorted, jittered, multi-line text.

Builds bottom-up — :class:`Character` -> :class:`Word` -> :class:`Phrase` —
applying per-level mesh deformations and producing, alongside the rendered image,
a per-pixel symbol ``label`` and a per-pixel word ``mask`` (kept aligned by
replaying the same distortion schedule on all three).
"""

# Pillow's type stubs type ``Image.getpixel`` as ``float | tuple | None`` and the
# layer images as ``Image | None`` until assigned in ``create_image``; this is
# pixel-manipulation glue where those flow through arithmetic/paste, so the
# stub-driven optional/argument noise is suppressed for this module only.
# pyright: reportOptionalMemberAccess=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false

from numpy.random import normal, permutation, randint
from PIL import Image, ImageDraw, ImageFont

from glyphloop.data.glossary import get_symbol_dict, get_word_dict
from glyphloop.data.synthesis.apply_deformations import (
    apply_random_distortions,
    batch_apply_distortion_schedule,
)
from glyphloop.data.synthesis.colors import value_to_grayscale, value_to_rgb
from glyphloop.data.synthesis.measurement import get_smallest_box_crop, smallest_box_crop
from glyphloop.data.synthesis.rng import randomize

symbols_legend = get_symbol_dict()
words_legend = get_word_dict()


def rgba_character_to_grayscale(image: Image.Image):
    """Flatten a rendered (colored) glyph to a binary-ish grayscale image."""
    X, Y = image.size
    flattened = Image.new(mode="L", size=image.size, color=0)
    for x in range(X):
        for y in range(Y):
            pixel_value = image.getpixel((x, y))
            flattened.putpixel((x, y), value=0 if pixel_value == (0, 0, 0) else 255)
    return flattened


class Character:
    def __init__(self, c, font, size, distortion_level):
        self.c = c
        self.image = None
        self.median_activation = None
        self.create_image(font, size, distortion_level)

    def size(self):
        return self.image.size

    def get_median_activation(self):
        if self.median_activation is None:
            assert self.image is not None, "you're trying to do stuff with an undefined image"
            y_activations = [
                sum(self.image.getpixel((x, y)) for x in range(self.image.size[0]))
                for y in range(self.image.size[1])
            ]
            self.median_activation = max(
                y
                for y in range(len(y_activations))
                if sum(y_activations[:y]) <= sum(y_activations) // 2
            )
        return self.median_activation

    def create_image(self, font, size, distortion_level):
        # 2*size is comfortably big enough to hold a glyph of point size `size`.
        self.image = Image.new(mode="RGB", size=(2 * size, 2 * size), color=(0, 0, 0))
        ImageDraw.Draw(self.image).text(
            xy=(0, 0),
            text=self.c,
            fill=(255, 255, 255),
            font=ImageFont.truetype(font, size),
            align="center",
        )
        self.image = smallest_box_crop(rgba_character_to_grayscale(self.image))
        self.image, _ = apply_random_distortions(self.image, distortion_level)
        self.image = smallest_box_crop(self.image)


class Word:
    def __init__(self, w, font, size, distortion_level):
        self.w = w
        self.image = None
        self.label = None
        self.mask = None
        self.create_image(font, size, distortion_level)

    def size(self):
        return self.image.size

    def create_image(self, font, size, distortion_level):
        self.chars = [Character(c, font, size, distortion_level) for c in self.w]

        max_width = sum(c.size()[0] for c in self.chars)
        max_height = max(c.size()[1] for c in self.chars)

        self.image = Image.new(mode="L", size=(3 * max_width, 4 * max_height), color=0)
        self.label = Image.new(mode="L", size=(3 * max_width, 4 * max_height), color=0)

        min_x = 0
        target_x = 0
        target_y = 2 * max_height
        for c in self.chars:
            cx, cy = c.size()
            median_activation = c.get_median_activation()

            x_align = max(min_x, int(normal(target_x, distortion_level)))
            y_align = int(normal(target_y, distortion_level)) - median_activation // 2

            for x in range(cx):
                for y in range(cy):
                    if (
                        0 <= x_align + x < self.image.size[0]
                        and 0 <= y_align + y < self.image.size[1]
                    ):
                        if self.image.getpixel((x_align + x, y_align + y)) < c.image.getpixel(
                            (x, y)
                        ):
                            self.image.putpixel(
                                (x_align + x, y_align + y), value=c.image.getpixel((x, y))
                            )
                        self.label.putpixel(
                            (x_align + x, y_align + y),
                            value=value_to_grayscale(symbols_legend[c.c]),
                        )

            # don't cross the previous glyph's center line, but aim near its right edge
            min_x = x_align + cx // 2
            target_x = x_align + cx

        # crop against the label (the largest of the three aligned layers)
        crop_box = get_smallest_box_crop(self.label)
        self.image = self.image.crop(crop_box)
        self.label = self.label.crop(crop_box)

        # a solid-color mask lets us record one distortion schedule and replay it
        # on the image + label so all three stay pixel-aligned.
        self.mask = Image.new(
            mode="RGB",
            size=self.image.size,
            color=value_to_rgb(words_legend.get(self.w, 1)),
        )
        self.mask, distortion_schedule = apply_random_distortions(self.mask, distortion_level)
        self.image, self.label = batch_apply_distortion_schedule(
            [self.image, self.label], distortion_schedule
        )

        crop_box = get_smallest_box_crop(self.mask)
        self.image = self.image.crop(crop_box)
        self.label = self.label.crop(crop_box)
        self.mask = self.mask.crop(crop_box)


class Phrase:
    """Arrange words across one-or-more lines, pick the layout whose aspect ratio
    best matches the target image, then scale, place and rotate the whole phrase."""

    def __init__(self, img_dim, p, font, size, distortion_level):
        self.p = p
        self.image = None
        self.label = None
        self.mask = None
        self.img_dim = img_dim
        self.create_image(font, size, distortion_level)

    def size(self):
        return self.image.size

    def create_image(self, font, size, distortion_level):
        self.words = [Word(w, font, size, distortion_level) for w in self.p.split(" ")]

        orientations = []
        for bits in range(2 ** (len(self.words) - 1)):
            x_align, x_max = 0, 0
            y_align, y_max = 0, 0
            placements = []
            for w in range(len(self.words)):
                placements.append((x_align, y_align))
                wx, wy = self.words[w].size()
                wx += randint(10, 50)  # spread things out a smidge
                wy += randint(10, 50)
                x_align += wx
                x_max = max(x_max, x_align)
                y_max = max(y_max, y_align + wy)
                if bits & (1 << w):
                    x_align = 0
                    y_align = y_max
            orientations.append(
                ((x_max / y_max) / (self.img_dim[0] / self.img_dim[1]), x_max, y_max, placements)
            )

        good_orientations = [o for o in orientations if o[0] < 1]
        orientation = (
            randomize(orientations)[0] if len(orientations) else randomize(good_orientations)[0]
        )
        x_max, y_max, placements = orientation[1:]

        self.image = Image.new(mode="L", size=(x_max, y_max), color=0)
        self.label = Image.new(mode="L", size=(x_max, y_max), color=0)
        self.mask = Image.new(mode="RGB", size=(x_max, y_max), color=(0, 0, 0))
        for i in range(len(placements)):
            self.image.paste(self.words[i].image, placements[i])
            self.label.paste(self.words[i].label, placements[i])
            self.mask.paste(self.words[i].mask, placements[i])

        if self.img_dim[0] < self.image.size[0] or self.img_dim[1] < self.image.size[1]:
            target_x = int(normal(self.img_dim[0] * 0.80, self.img_dim[0] * 0.05))
            target_y = int(normal(self.img_dim[1] * 0.80, self.img_dim[1] * 0.05))
            self.image = self.image.resize((target_x, target_y))
            self.label = self.label.resize((target_x, target_y))
            self.mask = self.mask.resize((target_x, target_y))

        final_image = Image.new(mode="L", size=self.img_dim, color=0)
        prefinal_label = Image.new(mode="L", size=self.img_dim, color=0)
        prefinal_mask = Image.new(mode="RGB", size=self.img_dim, color=(0, 0, 0))

        x_offset = randint(0, max(1, self.img_dim[0] - self.image.size[0]))
        y_offset = randint(0, max(1, self.img_dim[1] - self.image.size[1]))
        final_image.paste(self.image, (x_offset, y_offset))
        prefinal_label.paste(self.label, (x_offset, y_offset))
        prefinal_mask.paste(self.mask, (x_offset, y_offset))

        rotation = normal(0, distortion_level)
        self.image = final_image.rotate(rotation, resample=Image.Resampling.BICUBIC)
        self.label = prefinal_label.rotate(rotation)
        self.mask = prefinal_mask.rotate(rotation)


def generate_textground(img_dim, fonts, distortion_level, desired_phrase):
    phrase = Phrase(
        img_dim=img_dim,
        p=desired_phrase,
        font=permutation(fonts)[0],
        size=int(normal(50, 5)),
        distortion_level=distortion_level,
    )
    return phrase.p, phrase.mask, phrase.label, phrase.image
