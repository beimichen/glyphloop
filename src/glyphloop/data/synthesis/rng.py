"""Small randomization helpers (renamed from the original ``random.py`` so it no
longer shadows the standard-library :mod:`random` module)."""

from numpy.random import choice, permutation, randint, random


def randomize(array):
    """In-place Fisher-Yates-ish shuffle that also returns the array."""
    for i in range(len(array)):
        s = randint(len(array))
        array[i], array[s] = array[s], array[i]
    return array


def random_rgb():
    def r():
        return int(random() * 256)

    return r(), r(), r()


def random_rgba():
    def r():
        return int(random() * 256)

    return r(), r(), r(), 255


def random_center(img_dim):
    return int(random() * img_dim[0]), int(random() * img_dim[1])


def random_radius(img_dim, divisor=2):
    return int(random() * (img_dim[0] // divisor))


def sub_random_alphas(phrase):
    """Randomly substitute a random subset of alphabetic characters in a phrase."""
    alphas = [chr(x) for x in range(ord("a"), ord("z") + 1)]
    string_as_array = list(phrase)

    subbable = [x for x in range(len(string_as_array)) if string_as_array[x].isalpha()]
    for s in permutation(subbable)[: randint(1, len(subbable))]:
        string_as_array[s] = choice(alphas)

    return "".join(string_as_array)
