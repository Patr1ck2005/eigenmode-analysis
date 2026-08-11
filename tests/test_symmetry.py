from __future__ import annotations

import numpy as np

from eigenmode_analysis import (
    complete_C2_polarization,
    complete_C4_polarization,
    complete_C4_spectrum,
    complete_C6_polarization,
    geom_complete,
)


def _quadrant():
    coords = {"kx": np.array([0.0, 1.0, 2.0]), "ky": np.array([0.0, 1.0])}
    values = np.arange(6.0).reshape(3, 2)
    phi = values / 10.0
    chi = np.linspace(-0.5, 0.5, 6).reshape(3, 2)
    return coords, values, phi, chi


def test_geom_complete_preserves_x_first_axis_orientation():
    coords, values, _, _ = _quadrant()
    full_coords, full = geom_complete(coords, values, mode="xy")
    np.testing.assert_allclose(full_coords["kx"], [-2, -1, 0, 1, 2])
    np.testing.assert_allclose(full_coords["ky"], [-1, 0, 1])
    np.testing.assert_array_equal(
        full,
        np.array([[5, 4, 5], [3, 2, 3], [1, 0, 1], [3, 2, 3], [5, 4, 5]], dtype=float),
    )


def test_polarization_completion_keeps_c2_c4_c6_shapes_and_sign_rules():
    coords, _, phi, chi = _quadrant()
    c2_coords, c2_phi, c2_chi = complete_C2_polarization(coords, phi, chi, axis="x")
    c4_coords, c4_phi, c4_chi = complete_C4_polarization(coords, phi, chi)
    c6_coords, c6_phi, c6_chi = complete_C6_polarization(coords, phi, chi)
    assert c2_phi.shape == c2_chi.shape == (5, 2)
    assert c4_phi.shape == c4_chi.shape == (5, 3)
    assert c6_phi.shape == c6_chi.shape == (5, 3)
    np.testing.assert_allclose(c4_chi[0], [0.5, -0.3, -0.5])
    np.testing.assert_allclose(c6_chi[:, 0], [-0.5, -0.1, -0.3, 0.1, 0.5])
    assert set(c2_coords) == set(c4_coords) == set(c6_coords) == {"m1", "m2"}


def test_complete_c4_spectrum_rotates_complex_amplitude_with_sign_flip():
    coords = {"kx": np.array([0.0, 1.0]), "ky": np.array([0.0, 1.0])}
    values = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    full_coords, full = complete_C4_spectrum(coords, values)
    assert full.shape == (3, 3)
    np.testing.assert_array_equal(full_coords["kx"], [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(full_coords["ky"], [-1.0, 0.0, 1.0])
    np.testing.assert_equal(full[0, 0], values[1, 1])
    np.testing.assert_equal(full[0, 2], -values[1, 1])
