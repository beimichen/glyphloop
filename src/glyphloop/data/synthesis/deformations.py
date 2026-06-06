"""Mesh deformations for PIL's ``ImageOps.deform`` — lean, squash, perspective,
and corner-stretch transforms applied to glyphs, words and phrases."""


class TopLean:
    def __init__(self, mag):
        self.mag = mag

    def getmesh(self, img):
        X, Y = img.size
        return [((0, 0, X, Y), (int(self.mag * X), 0, 0, Y, X, Y, int((1 + self.mag) * X), 0))]


class RightLean:
    def __init__(self, mag):
        self.mag = mag

    def getmesh(self, img):
        X, Y = img.size
        return [((0, 0, X, Y), (0, 0, 0, Y, X, int((1 + self.mag) * Y), X, int(self.mag * Y)))]


class VerticalSquash:
    def __init__(self, mag):
        self.mag = mag / 2

    def getmesh(self, img):
        X, Y = img.size
        return [
            (
                (0, 0, X, Y),
                (
                    0,
                    int(-self.mag * Y),
                    0,
                    int((1 + self.mag) * Y),
                    X,
                    int((1 + self.mag) * Y),
                    X,
                    int(-self.mag * Y),
                ),
            )
        ]


class HorizontalSquash:
    def __init__(self, mag):
        self.mag = mag / 2

    def getmesh(self, img):
        X, Y = img.size
        return [
            (
                (0, 0, X, Y),
                (
                    int(-self.mag * X),
                    0,
                    int(-self.mag * X),
                    Y,
                    int((1 + self.mag) * X),
                    Y,
                    int((1 + self.mag) * X),
                    0,
                ),
            )
        ]


class LeftDiagonalSquash:
    def __init__(self, mag):
        self.mag = mag / 2

    def getmesh(self, img):
        X, Y = img.size
        return [
            (
                (0, 0, X, Y),
                (
                    int(-self.mag * X),
                    int(-self.mag * Y),
                    0,
                    Y,
                    int((1 + self.mag) * X),
                    int((1 + self.mag) * Y),
                    X,
                    0,
                ),
            )
        ]


class RightDiagonalSquash:
    def __init__(self, mag):
        self.mag = mag / 2

    def getmesh(self, img):
        X, Y = img.size
        return [
            (
                (0, 0, X, Y),
                (
                    0,
                    0,
                    int(-self.mag * X),
                    int((1 + self.mag) * Y),
                    X,
                    Y,
                    int((1 + self.mag) * X),
                    int(-self.mag * Y),
                ),
            )
        ]


class PerspectiveLeanTop:
    def __init__(self, mag):
        self.mag = mag / 2

    def getmesh(self, img):
        X, Y = img.size
        return [((0, 0, X, Y), (int(-self.mag * X), 0, 0, Y, X, Y, int((1 + self.mag) * X), 0))]


class PerspectiveLeanRight:
    def __init__(self, mag):
        self.mag = mag / 2

    def getmesh(self, img):
        X, Y = img.size
        return [((0, 0, X, Y), (0, 0, 0, Y, X, int((1 + self.mag) * Y), X, int(-self.mag * Y)))]


class ExtendUpperRightCorner:
    def __init__(self, xmag, ymag=None):
        self.xmag = xmag
        self.ymag = ymag if ymag is not None else xmag

    def getmesh(self, img):
        X, Y = img.size
        return [((0, 0, X, Y), (0, 0, 0, Y, X, Y, int((1 + self.xmag) * X), int(-self.ymag * Y)))]


class ExtendUpperLeftCorner:
    def __init__(self, xmag, ymag=None):
        self.xmag = xmag
        self.ymag = ymag if ymag is not None else xmag

    def getmesh(self, img):
        X, Y = img.size
        return [((0, 0, X, Y), (int(-self.xmag * X), int(-self.ymag * Y), 0, Y, X, Y, X, 0))]


class ExtendLowerLeftCorner:
    def __init__(self, xmag, ymag=None):
        self.xmag = xmag
        self.ymag = ymag if ymag is not None else xmag

    def getmesh(self, img):
        X, Y = img.size
        return [((0, 0, X, Y), (0, 0, int(-self.xmag * X), int((1 + self.ymag) * Y), X, Y, X, 0))]


class ExtendLowerRightCorner:
    def __init__(self, xmag, ymag=None):
        self.xmag = xmag
        self.ymag = ymag if ymag is not None else xmag

    def getmesh(self, img):
        X, Y = img.size
        return [
            ((0, 0, X, Y), (0, 0, 0, Y, int((1 + self.ymag) * X), int((1 + self.ymag) * Y), X, 0))
        ]


deformations = [
    TopLean,
    RightLean,
    VerticalSquash,
    HorizontalSquash,
    LeftDiagonalSquash,
    RightDiagonalSquash,
    PerspectiveLeanTop,
    PerspectiveLeanRight,
    ExtendUpperRightCorner,
    ExtendUpperLeftCorner,
    ExtendLowerRightCorner,
    ExtendLowerLeftCorner,
]
