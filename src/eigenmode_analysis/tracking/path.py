"""Explicit-anchor Hungarian tracking along a one-dimensional path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._common import (
    initialize_anchor,
    normalize_weights,
    result_dtype,
    transition_costs,
)
from .models import CandidateSet, TrackingConfig, TrackingResult, TransitionDiagnostic


_INF_COST = 1e15


@dataclass
class _BranchState:
    current: np.ndarray
    previous: np.ndarray
    anchor_values: np.ndarray
    anchor_coordinates: np.ndarray
    active: np.ndarray
    current_point: int
    previous_point: int | None


def _transition_costs(
    point: CandidateSet,
    state: _BranchState,
    coordinate: float,
    coordinates: np.ndarray,
    value_weights: np.ndarray,
    derivative_weights: np.ndarray,
    config: TrackingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = coordinate - coordinates[state.current_point]
    if delta == 0:
        raise ValueError("coordinates must be distinct along each tracking branch")

    previous_delta = None
    if state.previous_point is not None:
        previous_delta = coordinates[state.current_point] - coordinates[state.previous_point]

    costs, finite_candidates = transition_costs(
        point.features,
        state.current,
        state.previous,
        delta=delta,
        previous_delta=previous_delta,
        value_weights=value_weights,
        derivative_weights=derivative_weights,
        nan_cost_penalty=config.nan_cost_penalty,
    )

    outside_anchor_constraint = np.zeros_like(costs, dtype=bool)
    if config.rank_penalty > 0:
        candidate_values = np.real(point.features[:, config.anchor_feature])
        ranks = np.argsort(np.argsort(candidate_values))
        for track_id in range(state.current.shape[0]):
            costs[track_id] += (ranks - track_id) ** 2 * config.rank_penalty

    if config.anchor_window is not None:
        candidate_values = np.real(point.features[:, config.anchor_feature])
        for track_id, anchor_value in enumerate(state.anchor_values):
            if not np.isfinite(anchor_value):
                continue
            outside = np.abs(candidate_values - anchor_value) > config.anchor_window
            outside |= ~np.isfinite(candidate_values)
            outside_anchor_constraint[track_id] = outside
            costs[track_id, outside] += _INF_COST

    if config.anchor_slope is not None:
        candidate_values = np.real(point.features[:, config.anchor_feature])
        for track_id, (anchor_value, anchor_coordinate) in enumerate(
            zip(state.anchor_values, state.anchor_coordinates)
        ):
            if not np.isfinite(anchor_value) or not np.isfinite(anchor_coordinate):
                continue
            allowed_change = config.anchor_slope * abs(coordinate - anchor_coordinate)
            outside = np.abs(candidate_values - anchor_value) > allowed_change
            outside |= ~np.isfinite(candidate_values)
            outside_anchor_constraint[track_id] = outside
            costs[track_id, outside] += _INF_COST

    return costs, finite_candidates, outside_anchor_constraint


def _track_branch(
    sequence: Sequence[int],
    points: Sequence[CandidateSet],
    coordinates: np.ndarray,
    ids_out: list[list[Hashable | None]],
    features_out: np.ndarray,
    costs_out: np.ndarray,
    anchor_features: np.ndarray,
    value_weights: np.ndarray,
    derivative_weights: np.ndarray,
    config: TrackingConfig,
) -> list[TransitionDiagnostic]:
    if len(sequence) <= 1:
        return []

    anchor_index = int(sequence[0])
    state = _BranchState(
        current=anchor_features.copy(),
        previous=np.full_like(anchor_features, np.nan),
        anchor_values=np.real(anchor_features[:, config.anchor_feature]).copy(),
        anchor_coordinates=np.where(
            np.isfinite(anchor_features[:, config.anchor_feature]),
            coordinates[anchor_index],
            np.nan,
        ),
        active=np.all(np.isfinite(anchor_features), axis=1),
        current_point=anchor_index,
        previous_point=None,
    )
    diagnostics: list[TransitionDiagnostic] = []

    for point_index in sequence[1:]:
        point = points[point_index]
        previous_point_index = state.current_point
        next_features = np.full_like(anchor_features, np.nan)
        next_ids: list[Hashable | None] = [None] * anchor_features.shape[0]
        next_costs = np.full(anchor_features.shape[0], np.nan, dtype=float)
        assigned_tracks: set[int] = set()

        if point.n_candidates:
            matrix, finite_candidates, outside_window = _transition_costs(
                point,
                state,
                float(coordinates[point_index]),
                coordinates,
                value_weights,
                derivative_weights,
                config,
            )
            active_tracks = np.flatnonzero(state.active)
            rows, columns = linear_sum_assignment(matrix[active_tracks])
            for active_row, candidate_index in zip(rows.tolist(), columns.tolist()):
                track_id = int(active_tracks[active_row])
                assigned_tracks.add(track_id)
                candidate_id = point.ids[candidate_index]
                cost = float(matrix[track_id, candidate_index])
                reason = None
                if not finite_candidates[candidate_index]:
                    reason = "non_finite_candidate"
                elif outside_window[track_id, candidate_index]:
                    reason = (
                        "anchor_slope"
                        if config.anchor_slope is not None
                        else "anchor_window"
                    )

                accepted = reason is None
                if accepted:
                    next_ids[track_id] = candidate_id
                    next_features[track_id] = point.features[candidate_index]
                    next_costs[track_id] = cost
                    anchor_value = np.real(
                        point.features[candidate_index, config.anchor_feature]
                    )
                    if np.isfinite(anchor_value):
                        state.anchor_values[track_id] = anchor_value
                        state.anchor_coordinates[track_id] = coordinates[point_index]

                diagnostics.append(
                    TransitionDiagnostic(
                        point_index=int(point_index),
                        previous_point_index=int(previous_point_index),
                        track_id=int(track_id),
                        candidate_id=candidate_id,
                        cost=cost,
                        accepted=accepted,
                        reason=reason,
                    )
                )

        for track_id in range(anchor_features.shape[0]):
            if track_id in assigned_tracks:
                continue
            diagnostics.append(
                TransitionDiagnostic(
                    point_index=int(point_index),
                    previous_point_index=int(previous_point_index),
                    track_id=track_id,
                    candidate_id=None,
                    cost=None,
                    accepted=False,
                    reason="unassigned",
                )
            )

        ids_out[point_index] = next_ids
        features_out[point_index] = next_features
        costs_out[point_index] = next_costs
        state.previous = state.current
        state.previous_point = state.current_point
        state.current = next_features
        state.current_point = int(point_index)

    return diagnostics


def track_path(
    points: Sequence[CandidateSet],
    *,
    coordinates: Sequence[float] | None = None,
    anchor_index: int = 0,
    config: TrackingConfig | None = None,
) -> TrackingResult:
    """Track candidates from an explicit anchor in both path directions.

    Track labels are initialized by sorting the anchor candidates on the real
    part of ``config.anchor_feature``. The same anchor labels seed the forward
    and backward branches, which makes a Gamma-centered ``M-Gamma-X`` path an
    ordinary input rather than a project-specific special case.
    """

    points = tuple(points)
    if not points:
        raise ValueError("points cannot be empty")
    if not 0 <= anchor_index < len(points):
        raise ValueError("anchor_index is outside the path")

    n_features = points[0].n_features
    if n_features == 0:
        raise ValueError("candidate feature vectors cannot be empty")
    if any(point.n_features != n_features for point in points):
        raise ValueError("all path points must use the same feature dimension")

    config = config or TrackingConfig()
    if config.anchor_feature >= n_features:
        raise ValueError("anchor_feature is outside the feature vector")

    if coordinates is None:
        coordinate_array = np.arange(len(points), dtype=float)
    else:
        coordinate_array = np.asarray(coordinates, dtype=float)
        if coordinate_array.shape != (len(points),):
            raise ValueError("coordinates must contain one scalar per path point")
        if not np.all(np.isfinite(coordinate_array)):
            raise ValueError("coordinates must be finite")

    n_tracks = config.n_tracks or points[anchor_index].n_candidates
    if n_tracks <= 0:
        raise ValueError("the anchor has no candidates and n_tracks was not provided")

    value_weights = normalize_weights(config.value_weights, n_features, "value_weights")
    derivative_weights = normalize_weights(
        config.derivative_weights,
        n_features,
        "derivative_weights",
    )
    dtype = result_dtype(points)
    anchor_ids, anchor_features = initialize_anchor(
        points[anchor_index],
        n_tracks,
        config.anchor_feature,
        dtype,
    )

    ids_out: list[list[Hashable | None]] = [
        [None] * n_tracks for _ in range(len(points))
    ]
    ids_out[anchor_index] = anchor_ids
    features_out = np.full(
        (len(points), n_tracks, n_features),
        np.nan,
        dtype=dtype,
    )
    features_out[anchor_index] = anchor_features
    costs_out = np.full((len(points), n_tracks), np.nan, dtype=float)

    diagnostics: list[TransitionDiagnostic] = []
    diagnostics.extend(
        _track_branch(
            list(range(anchor_index, len(points))),
            points,
            coordinate_array,
            ids_out,
            features_out,
            costs_out,
            anchor_features,
            value_weights,
            derivative_weights,
            config,
        )
    )
    diagnostics.extend(
        _track_branch(
            list(range(anchor_index, -1, -1)),
            points,
            coordinate_array,
            ids_out,
            features_out,
            costs_out,
            anchor_features,
            value_weights,
            derivative_weights,
            config,
        )
    )

    return TrackingResult(
        anchor_index=int(anchor_index),
        coordinates=coordinate_array.copy(),
        candidate_ids=tuple(tuple(row) for row in ids_out),
        features=features_out,
        costs=costs_out,
        diagnostics=tuple(diagnostics),
    )
