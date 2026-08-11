from __future__ import annotations

import numpy as np
import pytest

from eigenmode_analysis import (
    bilinear_interpolate,
    bilinear_interpolate_angle_pi,
    resample_path_uniform,
    sample_fields_along_path,
)


def test_bilinear_interpolation_preserves_x_first_orientation():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 2.0])
    values = x[:, None] + 2.0 * y[None, :]
    result = bilinear_interpolate(x, y, values, np.array([0.5]), np.array([1.0]))
    np.testing.assert_allclose(result, [2.5])


def test_angle_interpolation_avoids_pi_branch_cut():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    values = np.array([[0.49 * np.pi, 0.49 * np.pi], [-0.49 * np.pi, -0.49 * np.pi]])
    result = bilinear_interpolate_angle_pi(x, y, values, np.array([0.5]), np.array([0.5]))
    assert abs(abs(float(result[0])) - 0.5 * np.pi) < 0.05


def test_path_resampling_and_field_sampling():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    _, arc = resample_path_uniform(path, npts=5)
    np.testing.assert_allclose(arc, np.linspace(0.0, 2.0, 5))
    fields = sample_fields_along_path(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        {"value": np.array([[0.0, 1.0], [1.0, 2.0]])},
        path,
        npts=3,
    )
    np.testing.assert_allclose(fields["value"], [0.0, 1.0, 2.0])


def test_invalid_path_is_rejected():
    with pytest.raises(ValueError):
        resample_path_uniform([[0.0, 0.0]])
