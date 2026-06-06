"""Mesh deformations return well-formed PIL deform meshes."""

from __future__ import annotations

from glyphloop.data.synthesis.deformations import deformations


class _FakeImg:
    size = (40, 20)


def test_every_deformation_returns_one_quad_mapping():
    img = _FakeImg()
    for cls in deformations:
        mesh = cls(0.2).getmesh(img)
        assert len(mesh) == 1
        box, quad = mesh[0]
        assert len(box) == 4  # (x0, y0, x1, y1) source box
        assert len(quad) == 8  # 4 destination corners
        assert all(isinstance(v, int) for v in quad)


def test_corner_deformations_accept_separate_x_y_magnitudes():
    from glyphloop.data.synthesis.deformations import ExtendUpperRightCorner

    mesh = ExtendUpperRightCorner(0.3, 0.1).getmesh(_FakeImg())
    assert len(mesh[0][1]) == 8
