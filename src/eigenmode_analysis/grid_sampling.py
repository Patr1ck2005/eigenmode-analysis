"""Schema-neutral regular-grid interpolation and path sampling primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def wrap_angle_pi(phi):
    """Wrap line-orientation angles to the interval ``[0, pi)``."""

    return np.asarray(phi) % np.pi


def find_cell_edges(grid):
    """Return left cell boundaries for a one-dimensional grid."""

    values = np.asarray(grid)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("grid must contain at least two one-dimensional points")
    delta = np.diff(values)
    if np.any(delta == 0):
        raise ValueError("grid points must be distinct")
    edges = np.empty_like(values)
    edges[1:] = (values[:-1] + values[1:]) / 2
    edges[0] = values[0] - delta[0] / 2
    return edges


def bilinear_interpolate(xgrid, ygrid, values, xq, yq):
    """Bilinearly interpolate an x-first scalar or trailing-field grid."""

    x_values = np.asarray(xgrid)
    y_values = np.asarray(ygrid)
    field = np.asarray(values)
    query_x = np.asarray(xq)
    query_y = np.asarray(yq)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("xgrid and ygrid must be one-dimensional")
    if field.shape[:2] != (x_values.size, y_values.size):
        raise ValueError(
            f"values shape {field.shape} must start with "
            f"({x_values.size}, {y_values.size})"
        )
    if query_x.shape != query_y.shape:
        raise ValueError("xq and yq must have equal shapes")

    x_edges = find_cell_edges(x_values)
    y_edges = find_cell_edges(y_values)
    i = np.clip(np.searchsorted(x_edges, query_x, side="right") - 1, 0, x_values.size - 2)
    j = np.clip(np.searchsorted(y_edges, query_y, side="right") - 1, 0, y_values.size - 2)
    tx = (query_x - x_values[i]) / (x_values[i + 1] - x_values[i] + 1e-12)
    ty = (query_y - y_values[j]) / (y_values[j + 1] - y_values[j] + 1e-12)
    f00 = field[i, j]
    f10 = field[i + 1, j]
    f01 = field[i, j + 1]
    f11 = field[i + 1, j + 1]
    return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11


def bilinear_interpolate_angle_pi(xgrid, ygrid, values, xq, yq):
    """Interpolate a line-orientation angle through ``cos(2 phi), sin(2 phi)``."""

    angle = wrap_angle_pi(values)
    cos2 = bilinear_interpolate(xgrid, ygrid, np.cos(2 * angle), xq, yq)
    sin2 = bilinear_interpolate(xgrid, ygrid, np.sin(2 * angle), xq, yq)
    return 0.5 * np.arctan2(sin2, cos2)


def resample_path_uniform(path_xy, step=None, npts=400):
    """Resample a polyline at uniform arc-length intervals."""

    points = np.asarray(path_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("path_xy must have shape (n, 2) with n >= 2")
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    arc = np.concatenate([[0.0], np.cumsum(distances)])
    if step is not None:
        if step <= 0:
            raise ValueError("step must be positive")
        count = max(2, int(math.ceil(arc[-1] / step)) + 1)
        target = np.linspace(0, arc[-1], count)
    else:
        if npts < 2:
            raise ValueError("npts must be at least 2")
        target = np.linspace(0, arc[-1], npts)
    sampled = np.column_stack((np.interp(target, arc, points[:, 0]), np.interp(target, arc, points[:, 1])))
    return sampled, target


def sample_fields_along_path(xgrid, ygrid, fields: Mapping[str, Any], path_xy, step=None, npts=400):
    """Sample scalar fields along a path with angle-aware interpolation."""

    sampled_path, arc = resample_path_uniform(path_xy, step=step, npts=npts)
    query_x, query_y = sampled_path[:, 0], sampled_path[:, 1]
    output: dict[str, Any] = {"s": arc, "x": query_x, "y": query_y}
    angle_names = {"phi", "phi_full", "angle", "polar_angle"}
    for name, values in fields.items():
        if name.lower() in angle_names:
            output[name] = bilinear_interpolate_angle_pi(xgrid, ygrid, values, query_x, query_y)
        else:
            output[name] = bilinear_interpolate(xgrid, ygrid, values, query_x, query_y)
    return output


__all__ = [
    "wrap_angle_pi",
    "find_cell_edges",
    "bilinear_interpolate",
    "bilinear_interpolate_angle_pi",
    "resample_path_uniform",
    "sample_fields_along_path",
]
