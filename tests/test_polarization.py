from __future__ import annotations

import numpy as np

from eigenmode_analysis import (
    ellipse_to_stokes,
    jones_from_stokes,
    normalize_stokes,
    poincare_points,
    stokes_from_jones,
    stokes_to_ellipse,
)


def test_ellipse_stokes_round_trip_preserves_signed_ratio():
    phi = np.array([-0.4, 0.0, 0.7])
    ratio = np.array([-0.5, 0.0, 0.25])
    stokes = ellipse_to_stokes(phi, ratio)
    phi_round, ratio_round = stokes_to_ellipse(*stokes)
    np.testing.assert_allclose(phi_round, phi)
    np.testing.assert_allclose(ratio_round, ratio)


def test_jones_stokes_round_trip_preserves_stokes_and_intensity():
    ex = np.array([1 + 0j, 1 / np.sqrt(2), 1 / np.sqrt(2)])
    ey = np.array([0 + 0j, 1j / np.sqrt(2), 1 / np.sqrt(2)])
    s0, s1, s2, s3 = stokes_from_jones(ex, ey)
    ex2, ey2 = jones_from_stokes(s1, s2, s3, s0)
    actual = stokes_from_jones(ex2, ey2)
    for expected, recovered in zip((s0, s1, s2, s3), actual):
        np.testing.assert_allclose(recovered, expected, atol=1e-12)


def test_zero_intensity_and_zero_stokes_have_explicit_policy():
    s0, s1, s2, s3 = stokes_from_jones([0], [0])
    assert s0[0] == 0
    assert np.isnan([s1[0], s2[0], s3[0]]).all()
    normalized = normalize_stokes([0], [0], [0], zero_policy="zero")
    assert all(component[0] == 0 for component in normalized)


def test_poincare_points_normalizes_and_drops_invalid_vectors():
    points = poincare_points([2, 0], [0, 0], [0, 0], flatten=True, drop_invalid=True)
    np.testing.assert_array_equal(points, [[1.0, 0.0, 0.0]])
