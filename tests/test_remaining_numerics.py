import numpy as np

from eigenmode_analysis import (
    angular_spectrum_propagate,
    angular_spectrum_xz_slice,
    extract_field_splits,
    fit_lineshape,
    lorentzian,
)


def test_lorentzian_fit_recovers_center_and_width():
    x = np.linspace(-2.0, 2.0, 201)
    y = lorentzian(x, 2.5, 0.25, 0.6, 0.1)
    result = fit_lineshape(x, y, model="lorentzian", output_samples=301)
    assert result.model == "lorentzian"
    assert np.isclose(result.params["x0"], 0.25, atol=1e-4)
    assert np.isclose(result.params["gamma"], 0.6, atol=1e-4)
    assert result.x_fit.shape == (301,)


def test_angular_spectrum_shapes_and_xz_slice():
    kx = np.linspace(-0.5, 0.5, 5)
    ky = np.linspace(-0.4, 0.4, 3)
    spectrum = np.zeros((len(ky), len(kx)), dtype=complex)
    spectrum[len(ky) // 2, len(kx) // 2] = 1.0
    propagated, (x, y, field) = angular_spectrum_propagate(kx, ky, spectrum, 0.2)
    assert propagated.shape == spectrum.shape
    assert field.shape == spectrum.shape
    assert x.shape == (len(kx),)
    assert y.shape == (len(ky),)

    x2, z, xz = angular_spectrum_xz_slice(kx, ky, spectrum, [0.0, 0.2])
    assert np.array_equal(x2, x)
    assert z.shape == (2,)
    assert xz.shape == (2, len(kx))


def test_field_regime_extraction_returns_partition_keys():
    x = np.linspace(-1.0, 1.0, 7)
    y = np.linspace(-1.0, 1.0, 7)
    s1 = np.broadcast_to(x[:, None], (len(x), len(y)))
    s2 = np.ones_like(s1)
    result = extract_field_splits(x, y, s1, s2)
    assert set(result) == {"phi0", "phi90", "uncertain"}
