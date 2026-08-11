from __future__ import annotations

import numpy as np
import pytest

from eigenmode_analysis import integrate_skyrmion_density, skyrmion_density


def test_density_uses_x_first_nonuniform_grid_and_correct_orientation():
    x = np.asarray([-2.0, -0.5, 0.0, 1.5, 4.0])
    y = np.asarray([-3.0, -1.0, -0.2, 0.5, 2.0, 5.0, 9.0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    density = skyrmion_density(X, Y, np.ones_like(X), x=x, y=y)
    np.testing.assert_allclose(density, 1.0, atol=1e-12)


def test_density_normalization_is_explicit_and_zero_vectors_become_nan():
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 4)
    X, Y = np.meshgrid(x, y, indexing="ij")
    raw = skyrmion_density(X, Y, np.ones_like(X), x=x, y=y, normalize=False)
    normalized = skyrmion_density(X, Y, np.ones_like(X), x=x, y=y, normalize=True)
    assert not np.allclose(raw, normalized, equal_nan=True)
    zero = skyrmion_density(
        np.zeros_like(X), np.zeros_like(X), np.zeros_like(X),
        x=x, y=y, normalize=True,
    )
    assert np.isnan(zero).all()


def test_integral_respects_coordinates_mask_and_invalid_data():
    x = np.asarray([0.0, 0.5, 2.0])
    y = np.asarray([-1.0, 1.0, 3.0, 4.0])
    density = np.full((len(x), len(y)), 2.0)
    expected = 2.0 * (x[-1] - x[0]) * (y[-1] - y[0]) / (4 * np.pi)
    assert integrate_skyrmion_density(density, x=x, y=y) == pytest.approx(expected)

    density[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        integrate_skyrmion_density(density, x=x, y=y)
    mask = np.ones_like(density, dtype=bool)
    mask[1, 1] = False
    assert np.isfinite(integrate_skyrmion_density(density, x=x, y=y, mask=mask))


def test_grid_contract_rejects_transpose_and_nonmonotonic_axes():
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 5)
    field = np.zeros((5, 3))
    with pytest.raises(ValueError, match="x-first"):
        skyrmion_density(field, field, field, x=x, y=y)
    with pytest.raises(ValueError, match="strictly increasing"):
        skyrmion_density(field.T, field.T, field.T, x=x[::-1], y=y)
