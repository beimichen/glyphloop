"""Crop images down to the bounding box of their non-zero content."""


def qsum(sorv):
    if isinstance(sorv, int):
        return sorv
    return sum(sorv)


def smallest_box_crop(image):
    """Crop to the content bounding box.

    Defensive against degenerate profiles (e.g. content touching an edge) that
    would otherwise produce a crossed crop box and make PIL raise — the original
    relied on a broad ``try/except`` around the whole pipeline to swallow that.
    """
    x_min, y_min, x_max, y_max = get_smallest_box_crop(image)
    x_max = max(x_max, x_min + 1)
    y_max = max(y_max, y_min + 1)
    return image.crop((x_min, y_min, x_max, y_max))


def get_smallest_box_crop(image):
    x_profile = [
        sum(qsum(image.getpixel((x, y))) for y in range(image.size[1]))
        for x in range(image.size[0])
    ]
    y_profile = [
        sum(qsum(image.getpixel((x, y))) for x in range(image.size[0]))
        for y in range(image.size[1])
    ]

    x_min = max([0] + [x for x in range(len(x_profile)) if sum(x_profile[:x]) == 0])
    x_max = min([image.size[0] - 1] + [x for x in range(len(x_profile)) if sum(x_profile[x:]) == 0])
    y_min = max([0] + [x for x in range(len(y_profile)) if sum(y_profile[:x]) == 0])
    y_max = min([image.size[1] - 1] + [x for x in range(len(y_profile)) if sum(y_profile[x:]) == 0])

    return x_min, y_min, x_max, y_max
