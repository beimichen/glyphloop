"""Validity checks for distorted glyphs (originally tab-indented; cleaned up)."""


def character_is_present_and_not_offscreen(image, size_xy=None):
    """True if a glyph survived a distortion: it has content and a clean border."""
    if size_xy is None:
        size_xy = image.size
    X, Y = size_xy

    all_values = set()
    border_values = set()
    for x in range(X):
        for y in range(Y):
            all_values.add(image.getpixel((x, y)))
            if x == 0 or x == X - 1 or y == 0 or y == Y - 1:
                border_values.add(image.getpixel((x, y)))

    return len(border_values) == 1 and len(all_values) >= 2


def character_does_not_overlap_an_existing_character(
    proposal, reference, size_xy, ctrl_value=(0, 0, 0, 0)
):
    X, Y = size_xy
    for x in range(X):
        for y in range(Y):
            if proposal.getpixel((x, y)) != ctrl_value and reference.getpixel((x, y)) != ctrl_value:
                return False
    return True
