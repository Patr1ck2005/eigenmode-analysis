"""Schema-neutral topological texture calculations on x-first grids."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from .polarization import normalize_stokes


def _grid(x, y, fields):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if len(x) < 2 or len(y) < 2:
        raise ValueError("topological texture grids require at least two points per axis")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite")
    if not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0):
        raise ValueError("x and y must be strictly increasing")
    expected = (len(x), len(y))
    arrays = tuple(np.asarray(field, dtype=float) for field in fields)
    if any(field.shape != expected for field in arrays):
        raise ValueError(f"fields must have x-first shape {expected}")
    return x, y, arrays


def skyrmion_density(s1, s2, s3, *, x, y, normalize: bool = False, edge_order: int = 2):
    """Return ``S dot (dS/dx cross dS/dy)`` on an explicit x-first grid."""

    x, y, fields = _grid(x, y, (s1, s2, s3))
    if normalize:
        fields = normalize_stokes(*fields)
    if edge_order not in {1, 2}:
        raise ValueError("edge_order must be 1 or 2")
    effective_edge_order = min(edge_order, 1 if min(len(x), len(y)) < 3 else 2)
    gradients = [
        np.gradient(field, x, y, axis=(0, 1), edge_order=effective_edge_order)
        for field in fields
    ]
    dsdx = np.stack([gradient[0] for gradient in gradients], axis=-1)
    dsdy = np.stack([gradient[1] for gradient in gradients], axis=-1)
    vectors = np.stack(fields, axis=-1)
    return np.einsum("...i,...i->...", vectors, np.cross(dsdx, dsdy))


def integrate_skyrmion_density(density, *, x, y, mask=None):
    """Integrate a density over x/y and divide by ``4*pi``."""

    x, y, (density,) = _grid(x, y, (density,))
    if mask is None:
        mask = np.ones_like(density, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != density.shape:
            raise ValueError("mask must match density shape")
    if np.any(~np.isfinite(density[mask])):
        raise ValueError("density contains non-finite values inside the integration mask")
    selected = np.where(mask, density, 0.0)
    integral_y = trapezoid(selected, y, axis=1)
    return float(trapezoid(integral_y, x, axis=0) / (4.0 * np.pi))
