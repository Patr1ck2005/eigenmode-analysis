"""Symmetry completion primitives for rectangular momentum-space data.

Scalar fields are shaped ``(len(kx), len(ky))`` and coordinates are
one-dimensional increasing axes that include zero. No pandas, plotting, or
consumer schema is involved here.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _wrap_angle_pi(phi):
    return np.asarray(phi) % np.pi


def _axis_coords(arr):
    arr = np.asarray(arr)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("symmetry axes must be non-empty 1D arrays")
    left = -arr[1:][::-1] if np.isclose(arr[0], 0) else -arr[::-1]
    return np.concatenate([left, arr], axis=0)


def _angle_mirror(phi):
    return _wrap_angle_pi(-_wrap_angle_pi(phi))


def _mirror_xy_data(values):
    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError("2D symmetry completion requires at least 2 dimensions")
    mirrored_x = np.concatenate([np.flip(values[1:], axis=0), values], axis=0)
    return np.concatenate([np.flip(mirrored_x[:, 1:], axis=1), mirrored_x], axis=1)


def _coord_keys(coords: Mapping[str, np.ndarray]):
    keys = list(coords)
    if {"kx", "ky"}.issubset(coords):
        return "kx", "ky"
    if len(keys) < 2:
        raise ValueError("coords must contain at least two axes")
    return keys[0], keys[1]


def _validate_quadrant(coords, values):
    xk, yk = _coord_keys(coords)
    x = np.asarray(coords[xk])
    y = np.asarray(coords[yk])
    values = np.asarray(values)
    if x.ndim != y.ndim or x.ndim != 1:
        raise ValueError("symmetry completion requires 1D coordinate axes")
    if x.size == 0 or y.size == 0 or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError("symmetry axes must be strictly increasing")
    if not np.isclose(x[0], 0) or not np.isclose(y[0], 0):
        raise ValueError("symmetry completion requires axes beginning at zero")
    if values.shape[:2] != (x.size, y.size):
        raise ValueError(
            f"values leading shape must be {(x.size, y.size)}, got {values.shape[:2]}"
        )
    return xk, yk, x, y, values


def geom_complete(coords, values, mode="xy"):
    """Complete a scalar/complex field by geometric axis reflection.

    ``mode`` accepts ``"x"``, ``"y"``, ``"xy"`` and the historical ``"C4"``
    alias.  The returned coordinate mapping keeps the input axis names.
    """

    xk, yk, x, y, values = _validate_quadrant(coords, values)
    if mode == "xy" or mode == "C4":
        x_full, y_full = _axis_coords(x), _axis_coords(y)
        values_full = _mirror_xy_data(values)
    elif mode == "x":
        x_full, y_full = _axis_coords(x), y
        values_full = np.concatenate([np.flip(values[1:], axis=0), values], axis=0)
    elif mode == "y":
        x_full, y_full = x, _axis_coords(y)
        values_full = np.concatenate([np.flip(values[:, 1:], axis=1), values], axis=1)
    else:
        raise ValueError(f"unknown symmetry completion mode: {mode}")
    result_coords = dict(coords)
    result_coords[xk], result_coords[yk] = x_full, y_full
    return result_coords, values_full


def _mirror_phi_x(phi):
    return _angle_mirror(np.flip(phi[:, 1:], axis=1))


def _mirror_phi_y(phi):
    return _angle_mirror(np.flip(phi[1:], axis=0))


def _mirror_phi_chi_x(phi, chi):
    return _mirror_phi_x(phi), -np.flip(chi[:, 1:], axis=1)


def _mirror_phi_chi_y(phi, chi):
    return _mirror_phi_y(phi), -np.flip(chi[1:], axis=0)


def _mirror_even_chi_x(chi):
    return np.flip(chi[:, 1:], axis=1)


def _mirror_even_chi_y(chi):
    return np.flip(chi[1:], axis=0)


def _mirror_odd_chi_y(chi):
    return -np.flip(chi[1:], axis=0)


def complete_C2_polarization(coords, phi, chi, axis="x"):
    """Complete polarization angle/ellipticity across one mirror axis."""

    _, _, x, y, values = _validate_quadrant(coords, phi)
    chi = np.asarray(chi)
    if chi.shape[:2] != values.shape[:2]:
        raise ValueError("phi and chi leading shapes must match")
    phi = _wrap_angle_pi(values)
    if axis == "x":
        x_full, y_full = _axis_coords(x), y
        phi_m, chi_m = _mirror_phi_chi_y(phi, chi)
        phi_full = np.concatenate([phi_m, phi], axis=0)
        chi_full = np.concatenate([chi_m, chi], axis=0)
    elif axis == "y":
        x_full, y_full = x, _axis_coords(y)
        phi_m, chi_m = _mirror_phi_chi_x(phi, chi)
        phi_full = np.concatenate([phi_m, phi], axis=1)
        chi_full = np.concatenate([chi_m, chi], axis=1)
    else:
        raise ValueError("axis must be 'x' or 'y'")
    return {"m1": x_full, "m2": y_full}, _wrap_angle_pi(phi_full), chi_full


def complete_C4_polarization(coords, phi, chi):
    """Complete C4 polarization using A's mirror/sign convention."""

    _, _, x, y, values = _validate_quadrant(coords, phi)
    chi = np.asarray(chi)
    if chi.shape[:2] != values.shape[:2]:
        raise ValueError("phi and chi leading shapes must match")
    phi = _wrap_angle_pi(values)
    x_full, y_full = _axis_coords(x), _axis_coords(y)
    phi_r, chi_r = _mirror_phi_chi_x(phi, chi)
    phi_r = np.concatenate([phi_r, phi], axis=1)
    chi_r = np.concatenate([chi_r, chi], axis=1)
    phi_l, chi_l = _mirror_phi_chi_y(phi_r, chi_r)
    return (
        {"m1": x_full, "m2": y_full},
        _wrap_angle_pi(np.concatenate([phi_l, phi_r], axis=0)),
        np.concatenate([chi_l, chi_r], axis=0),
    )


def complete_C6_polarization(coords, phi, chi):
    """Complete C6 polarization using the established even/odd chi rules."""

    _, _, x, y, values = _validate_quadrant(coords, phi)
    chi = np.asarray(chi)
    if chi.shape[:2] != values.shape[:2]:
        raise ValueError("phi and chi leading shapes must match")
    phi = _wrap_angle_pi(values)
    x_full, y_full = _axis_coords(x), _axis_coords(y)
    phi_r = _mirror_phi_x(phi)
    chi_r = _mirror_even_chi_x(chi)
    phi_r = np.concatenate([phi_r, phi], axis=1)
    chi_r = np.concatenate([chi_r, chi], axis=1)
    phi_l = _mirror_phi_y(phi_r)
    chi_l = _mirror_odd_chi_y(chi_r)
    return (
        {"m1": x_full, "m2": y_full},
        _wrap_angle_pi(np.concatenate([phi_l, phi_r], axis=0)),
        np.concatenate([chi_l, chi_r], axis=0),
    )


def complete_C4_spectrum(coords, complex_amplitude, eps=1e-9):
    """Complete a first-quadrant complex spectrum under C4 rotation."""

    xk, yk = _coord_keys(coords)
    kx, ky = np.asarray(coords[xk]), np.asarray(coords[yk])
    amplitude = np.asarray(complex_amplitude)
    if kx.ndim == ky.ndim == 1:
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        if amplitude.shape != KX.shape:
            amplitude = amplitude.reshape(KX.shape)
    elif kx.ndim == ky.ndim == 2:
        KX, KY = kx, ky
        if amplitude.shape != KX.shape:
            amplitude = amplitude.reshape(KX.shape)
    else:
        raise ValueError("kx and ky must both be 1D or both be 2D")
    mask = (KX >= -eps) & (KY >= -eps)
    x, y, values = KX[mask].ravel(), KY[mask].ravel(), amplitude[mask].ravel()
    x = np.where(np.abs(x) < eps, 0.0, x)
    y = np.where(np.abs(y) < eps, 0.0, y)
    rotations = ((x, y, values), (-y, x, -values), (-x, -y, values), (y, -x, -values))

    def clamp(values):
        values = np.asarray(values, dtype=float).copy()
        values[np.abs(values) < eps] = 0.0
        return np.round(values, 12)

    all_x = np.concatenate([clamp(item[0]) for item in rotations])
    all_y = np.concatenate([clamp(item[1]) for item in rotations])
    axes_x, axes_y = np.unique(all_x), np.unique(all_y)
    if axes_x.size == 0 or axes_x[0] > 0:
        axes_x = np.insert(axes_x, 0, 0.0)
    if axes_y.size == 0 or axes_y[0] > 0:
        axes_y = np.insert(axes_y, 0, 0.0)
    axes_x.sort(); axes_y.sort()
    output = np.full((axes_y.size, axes_x.size), np.nan + 1j * np.nan, dtype=complex)
    filled = np.zeros(output.shape, dtype=bool)
    dx = np.min(np.diff(axes_x)) if axes_x.size > 1 else 1e-9
    dy = np.min(np.diff(axes_y)) if axes_y.size > 1 else 1e-9

    def place(px, py, pv):
        px, py = clamp(px), clamp(py)
        ix = np.clip(np.searchsorted(axes_x, px), 0, axes_x.size - 1)
        iy = np.clip(np.searchsorted(axes_y, py), 0, axes_y.size - 1)
        for xi, yi, value, x_value, y_value in zip(ix, iy, pv, px, py):
            if abs(axes_x[xi] - x_value) <= 0.25 * dx and abs(axes_y[yi] - y_value) <= 0.25 * dy:
                output[yi, xi] = 0.5 * (output[yi, xi] + value) if filled[yi, xi] else value
                filled[yi, xi] = True

    for rotation in rotations:
        place(*rotation)

    x0 = int(np.flatnonzero(np.isclose(axes_x, 0.0, atol=eps))[0])
    y0 = int(np.flatnonzero(np.isclose(axes_y, 0.0, atol=eps))[0])
    for row in range(output.shape[0]):
        if np.isnan(output[row, x0]):
            candidates = np.flatnonzero(~np.isnan(output[row]))
            if candidates.size:
                output[row, x0] = output[row, candidates[0]]
    for column in range(output.shape[1]):
        if np.isnan(output[y0, column]):
            candidates = np.flatnonzero(~np.isnan(output[:, column]))
            if candidates.size:
                output[y0, column] = output[candidates[0], column]
    return {xk: axes_x, yk: axes_y}, output
