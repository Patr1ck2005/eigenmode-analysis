"""Internal numerical helpers shared by path and graph tracking."""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np

from .models import CandidateSet


def normalize_weights(value, n_features: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(n_features, float(array))
    if array.shape != (n_features,):
        raise ValueError(f"{name} must be a scalar or contain one value per feature")
    if np.any(array < 0) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def result_dtype(points: Sequence[CandidateSet]):
    populated = [point.features.dtype for point in points if point.features.size]
    return np.result_type(*populated, float) if populated else float


def initialize_anchor(
    point: CandidateSet,
    n_tracks: int,
    sort_feature: int,
    dtype,
) -> tuple[list[Hashable | None], np.ndarray]:
    ids: list[Hashable | None] = [None] * n_tracks
    features = np.full((n_tracks, point.n_features), np.nan, dtype=dtype)
    if point.n_candidates == 0:
        return ids, features

    finite = np.all(np.isfinite(point.features), axis=1)
    indices = [index for index in range(point.n_candidates) if finite[index]]
    indices.sort(
        key=lambda index: float(np.real(point.features[index, sort_feature]))
    )
    for track_id, candidate_index in enumerate(indices[:n_tracks]):
        ids[track_id] = point.ids[candidate_index]
        features[track_id] = point.features[candidate_index]
    return ids, features


def transition_costs(
    candidate_features: np.ndarray,
    current: np.ndarray,
    previous: np.ndarray,
    *,
    delta: float,
    previous_delta: float | None,
    value_weights: np.ndarray,
    derivative_weights: np.ndarray,
    nan_cost_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-track candidate costs and the finite-candidate mask."""

    n_tracks = current.shape[0]
    costs = np.full(
        (n_tracks, candidate_features.shape[0]),
        nan_cost_penalty * 2,
        dtype=float,
    )
    finite_candidates = np.all(np.isfinite(candidate_features), axis=1)

    for track_id in range(n_tracks):
        current_features = current[track_id]
        if not np.all(np.isfinite(current_features)):
            costs[track_id, finite_candidates] = nan_cost_penalty
            continue

        derivative = np.zeros(
            candidate_features.shape[1],
            dtype=current_features.dtype,
        )
        previous_features = previous[track_id]
        if (
            previous_delta is not None
            and previous_delta != 0
            and np.all(np.isfinite(previous_features))
        ):
            derivative = (current_features - previous_features) / previous_delta

        differences = candidate_features - current_features
        value_cost = np.sum(np.abs(differences) ** 2 * value_weights, axis=1)
        predicted_differences = differences - derivative * delta
        derivative_cost = np.sum(
            np.abs(predicted_differences) ** 2 * derivative_weights,
            axis=1,
        )
        row_cost = value_cost + derivative_cost
        costs[track_id] = np.where(
            finite_candidates & np.isfinite(row_cost),
            row_cost,
            nan_cost_penalty * 2,
        )

    return costs, finite_candidates
