"""Jones, Stokes, ellipse, and Poincare numerical conversions.

The default handedness convention is
``S3 = -2 Im(Ex conj(Ey))``. All normalized outputs use NaN for zero or
non-finite intensity instead of emitting divide-by-zero warnings.
"""

from __future__ import annotations

import numpy as np


def _safe_normalize(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    output = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=float)
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    np.divide(numerator, denominator, out=output, where=valid)
    return output


def normalize_stokes(s1, s2, s3, *, zero_policy: str = "nan"):
    """Normalize Stokes vectors to the Poincare sphere."""

    s1, s2, s3 = np.broadcast_arrays(
        np.asarray(s1, dtype=float),
        np.asarray(s2, dtype=float),
        np.asarray(s3, dtype=float),
    )
    norm = np.sqrt(s1 * s1 + s2 * s2 + s3 * s3)
    if zero_policy not in {"nan", "zero"}:
        raise ValueError("zero_policy must be 'nan' or 'zero'")
    out = tuple(_safe_normalize(component, norm) for component in (s1, s2, s3))
    if zero_policy == "zero":
        out = tuple(np.nan_to_num(component, nan=0.0) for component in out)
    return out


def ellipse_to_stokes(phi, tan_chi):
    """Convert ellipse orientation and signed axis ratio to normalized Stokes."""

    phi, tan_chi = np.broadcast_arrays(np.asarray(phi), np.asarray(tan_chi))
    denominator = 1.0 + tan_chi * tan_chi
    linear = (1.0 - tan_chi * tan_chi) / denominator
    return (
        np.cos(2.0 * phi) * linear,
        np.sin(2.0 * phi) * linear,
        2.0 * tan_chi / denominator,
    )


def stokes_to_ellipse(s1, s2, s3, *, normalize: bool = True):
    """Return orientation ``phi`` and signed ``tan(chi)`` from Stokes vectors."""

    if normalize:
        s1, s2, s3 = normalize_stokes(s1, s2, s3)
    else:
        s1, s2, s3 = np.broadcast_arrays(s1, s2, s3)
    phi = 0.5 * np.arctan2(s2, s1)
    chi = 0.5 * np.arcsin(np.clip(s3, -1.0, 1.0))
    return phi, np.tan(chi)


def stokes_from_jones(ex, ey, *, normalize: bool = True):
    """Compute Stokes parameters from Jones components using A's S3 sign."""

    ex, ey = np.broadcast_arrays(np.asarray(ex, dtype=complex), np.asarray(ey, dtype=complex))
    s0 = np.abs(ex) ** 2 + np.abs(ey) ** 2
    s1 = np.abs(ex) ** 2 - np.abs(ey) ** 2
    cross = ex * np.conj(ey)
    s2 = 2.0 * np.real(cross)
    s3 = -2.0 * np.imag(cross)
    if normalize:
        s1, s2, s3 = (_safe_normalize(value, s0) for value in (s1, s2, s3))
    return s0, s1, s2, s3


def jones_from_stokes(s1, s2, s3, s0=None, *, normalize: bool = True):
    """Construct one fully-polarized Jones representative from Stokes data."""

    if normalize:
        s1, s2, s3 = normalize_stokes(s1, s2, s3)
    else:
        s1, s2, s3 = np.broadcast_arrays(s1, s2, s3)
    if s0 is None:
        s0 = np.ones_like(s1, dtype=float)
    else:
        s0 = np.broadcast_to(np.asarray(s0, dtype=float), np.shape(s1))
    psi = 0.5 * np.arctan2(s2, s1)
    chi = 0.5 * np.arcsin(np.clip(s3, -1.0, 1.0))
    major, minor = np.cos(chi), np.sin(chi)
    amplitude = np.sqrt(np.where(s0 >= 0, s0, np.nan))
    ex = amplitude * (major * np.cos(psi) - 1j * minor * np.sin(psi))
    ey = amplitude * (major * np.sin(psi) + 1j * minor * np.cos(psi))
    return ex, ey


def poincare_points(s1, s2, s3, *, flatten: bool = False, drop_invalid: bool = False):
    """Return normalized Stokes vectors with a final coordinate dimension."""

    points = np.stack(normalize_stokes(s1, s2, s3), axis=-1)
    if flatten:
        points = points.reshape(-1, 3)
    if drop_invalid:
        if not flatten:
            raise ValueError("drop_invalid requires flatten=True")
        points = points[np.all(np.isfinite(points), axis=1)]
    return points
