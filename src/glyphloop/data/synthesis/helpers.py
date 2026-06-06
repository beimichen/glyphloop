"""Geometric helpers for the synthesis layers."""

from math import cos, radians, sin

from numpy.random import random


def get_polygon(center, radius, points):
    """Return the vertices of a regular ``points``-gon at a random orientation."""

    def get_point(center, angle, distance):
        return (
            center[0] + distance * cos(radians(angle)),
            center[1] + distance * sin(radians(angle)),
        )

    starting_angle = random() * 360
    return [get_point(center, starting_angle + x * (360 / points), radius) for x in range(points)]
