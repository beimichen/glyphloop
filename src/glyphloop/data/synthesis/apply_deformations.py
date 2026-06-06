"""Apply (and replay) random mesh deformations with feasibility checking."""

from numpy.random import normal, permutation
from PIL import Image, ImageOps

from glyphloop.data.synthesis.deformations import deformations
from glyphloop.data.synthesis.feasibility import character_is_present_and_not_offscreen


def _padded_collector(image):
    """A 3x canvas with ``image`` centered, so deformations have room to push pixels."""
    collector = Image.new(
        mode=image.mode,
        size=(3 * image.size[0], 3 * image.size[1]),
        color=0 if image.mode == "L" else (0, 0, 0),
    )
    collector.paste(
        image,
        (
            collector.size[0] // 2 - image.size[0] // 2,
            collector.size[1] // 2 - image.size[1] // 2,
        ),
    )
    return collector


def apply_random_distortions(image, num_distortions, distortion_sigma=0.1):
    """Apply ``num_distortions`` random deformations, keeping only feasible ones.

    Returns the distorted image and the accepted ``(deformation, magnitude)``
    schedule, so the exact same distortions can be replayed on aligned images
    (the glyph image, its symbol-label, and its word-mask) via
    :func:`apply_distortion_schedule`.
    """
    collector = _padded_collector(image)

    accepted_distortions = []
    while len(accepted_distortions) < num_distortions:
        function = permutation(deformations)[0]
        argument = normal(0, distortion_sigma)
        proposal = ImageOps.deform(collector, function(argument))
        if character_is_present_and_not_offscreen(proposal):
            collector = proposal
            accepted_distortions.append((function, argument))

    return collector, accepted_distortions


def apply_distortion_schedule(image, distortion_schedule):
    collector = _padded_collector(image)
    for function, argument in distortion_schedule:
        collector = ImageOps.deform(collector, function(argument))
    return collector


def batch_apply_distortion_schedule(images, distortion_schedule):
    return [apply_distortion_schedule(image, distortion_schedule) for image in images]
