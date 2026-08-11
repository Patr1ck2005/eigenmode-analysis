"""Candidate tracking over an explicit directed acyclic traversal graph."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Hashable

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._common import (
    initialize_anchor,
    normalize_weights,
    result_dtype,
    transition_costs,
)
from .models import (
    CandidateSet,
    GraphConflictDiagnostic,
    GraphTrackingResult,
    GraphTransitionDiagnostic,
    TrackingConfig,
    TraversalEdge,
)


_INF_COST = 1e15


def rectilinear_traversal(
    shape: Sequence[int],
    deltas: Sequence[float],
) -> tuple[tuple[tuple[int, ...], ...], tuple[TraversalEdge, ...]]:
    """Return the explicit positive-axis DAG used by an origin-grown grid."""

    shape = tuple(int(size) for size in shape)
    deltas = tuple(float(delta) for delta in deltas)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("shape must contain positive axis sizes")
    if len(deltas) != len(shape):
        raise ValueError("deltas must contain one step per grid axis")
    if any(delta == 0 or not np.isfinite(delta) for delta in deltas):
        raise ValueError("grid deltas must be finite and non-zero")

    node_order = tuple(np.ndindex(*shape))
    edges = []
    for target in node_order:
        for axis in range(len(shape)):
            if target[axis] == 0:
                continue
            source = list(target)
            source[axis] -= 1
            edges.append(
                TraversalEdge(
                    source=tuple(source),
                    target=target,
                    axis=axis,
                    step=abs(deltas[axis]),
                )
            )
    return node_order, tuple(edges)


def _validate_graph(
    points: Mapping[Hashable, CandidateSet],
    node_order: Sequence[Hashable],
    edges: Sequence[TraversalEdge],
    anchor_id: Hashable,
) -> tuple[
    tuple[Hashable, ...],
    dict[Hashable, int],
    dict[Hashable, tuple[TraversalEdge, ...]],
]:
    node_ids = tuple(node_order)
    if not node_ids:
        raise ValueError("node_order cannot be empty")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node_order must contain unique node ids")
    if node_ids[0] != anchor_id:
        raise ValueError("anchor_id must be the first node in node_order")
    if set(points) != set(node_ids):
        raise ValueError("points keys must match node_order exactly")

    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    incoming: dict[Hashable, list[TraversalEdge]] = defaultdict(list)
    seen_edges = set()
    for edge in edges:
        if edge.source not in positions or edge.target not in positions:
            raise ValueError("every traversal edge endpoint must exist in node_order")
        if positions[edge.source] >= positions[edge.target]:
            raise ValueError("every traversal edge must point forward in node_order")
        edge_key = (edge.source, edge.target, edge.axis)
        if edge_key in seen_edges:
            raise ValueError("duplicate traversal edge")
        seen_edges.add(edge_key)
        incoming[edge.target].append(edge)

    for node_id, node_edges in incoming.items():
        by_axis: dict[Hashable, int] = defaultdict(int)
        for edge in node_edges:
            if edge.axis is not None:
                by_axis[edge.axis] += 1
        if any(count > 1 for count in by_axis.values()):
            raise ValueError(
                f"node {node_id!r} has multiple incoming edges on the same axis"
            )

    ordered_incoming = {
        node_id: tuple(
            sorted(
                node_edges,
                key=lambda edge: (
                    positions[edge.source],
                    repr(edge.axis),
                    repr(edge.target),
                ),
            )
        )
        for node_id, node_edges in incoming.items()
    }
    return node_ids, positions, ordered_incoming


def _edge_previous(
    edge: TraversalEdge,
    incoming: Mapping[Hashable, tuple[TraversalEdge, ...]],
) -> TraversalEdge | None:
    if edge.axis is None:
        return None
    matches = [
        previous
        for previous in incoming.get(edge.source, ())
        if previous.axis == edge.axis
    ]
    if len(matches) > 1:
        raise ValueError(
            f"node {edge.source!r} has ambiguous predecessors on axis {edge.axis!r}"
        )
    return matches[0] if matches else None


def _accepted_edge_proposals(
    point: CandidateSet,
    matrix: np.ndarray,
    finite_candidates: np.ndarray,
    outside_window: np.ndarray,
    active_tracks: np.ndarray,
    informative_tracks: np.ndarray,
) -> dict[int, Hashable | None]:
    proposals = {int(track_id): None for track_id in informative_tracks}
    if not point.n_candidates or not informative_tracks.size:
        return proposals
    rows, columns = linear_sum_assignment(matrix[informative_tracks])
    for row, candidate_index in zip(rows.tolist(), columns.tolist()):
        track_id = int(informative_tracks[row])
        if (
            finite_candidates[candidate_index]
            and not outside_window[track_id, candidate_index]
            and active_tracks[track_id]
        ):
            proposals[track_id] = point.ids[candidate_index]
    return proposals


def track_graph(
    points: Mapping[Hashable, CandidateSet],
    *,
    node_order: Sequence[Hashable],
    edges: Sequence[TraversalEdge],
    anchor_id: Hashable,
    config: TrackingConfig | None = None,
) -> GraphTrackingResult:
    """Track candidates over a caller-defined DAG.

    Incoming edge costs are summed at merge nodes. Each informative incoming
    edge also proposes an assignment independently; disagreements are returned
    as conflict diagnostics instead of being hidden by traversal order.
    """

    config = config or TrackingConfig()
    if config.anchor_slope is not None:
        raise NotImplementedError(
            "anchor_slope requires a graph path-distance policy; use anchor_window"
        )
    node_ids, positions, incoming = _validate_graph(
        points,
        node_order,
        edges,
        anchor_id,
    )
    point_sequence = tuple(points[node_id] for node_id in node_ids)
    n_features = point_sequence[0].n_features
    if n_features == 0:
        raise ValueError("candidate feature vectors cannot be empty")
    if any(point.n_features != n_features for point in point_sequence):
        raise ValueError("all graph points must use the same feature dimension")
    if config.anchor_feature >= n_features:
        raise ValueError("anchor_feature is outside the feature vector")

    anchor_point = points[anchor_id]
    n_tracks = config.n_tracks or anchor_point.n_candidates
    if n_tracks <= 0:
        raise ValueError("the anchor has no candidates and n_tracks was not provided")
    value_weights = normalize_weights(
        config.value_weights,
        n_features,
        "value_weights",
    )
    derivative_weights = normalize_weights(
        config.derivative_weights,
        n_features,
        "derivative_weights",
    )
    dtype = result_dtype(point_sequence)
    anchor_ids, anchor_features = initialize_anchor(
        anchor_point,
        n_tracks,
        config.anchor_feature,
        dtype,
    )
    active_tracks = np.all(np.isfinite(anchor_features), axis=1)

    ids_out: list[list[Hashable | None]] = [
        [None] * n_tracks for _ in node_ids
    ]
    features_out = np.full(
        (len(node_ids), n_tracks, n_features),
        np.nan,
        dtype=dtype,
    )
    costs_out = np.full((len(node_ids), n_tracks), np.nan, dtype=float)
    anchor_memory = np.full((len(node_ids), n_tracks), np.nan, dtype=float)
    reachable = np.zeros(len(node_ids), dtype=bool)

    ids_out[0] = anchor_ids
    features_out[0] = anchor_features
    anchor_memory[0] = np.real(anchor_features[:, config.anchor_feature])
    reachable[0] = True

    diagnostics: list[GraphTransitionDiagnostic] = []
    conflicts: list[GraphConflictDiagnostic] = []
    unreachable_nodes: list[Hashable] = []

    for target_id in node_ids[1:]:
        target_position = positions[target_id]
        point = points[target_id]
        target_edges = tuple(
            edge
            for edge in incoming.get(target_id, ())
            if reachable[positions[edge.source]]
        )
        incoming_sources = tuple(edge.source for edge in target_edges)
        if not target_edges:
            unreachable_nodes.append(target_id)
            for track_id in range(n_tracks):
                diagnostics.append(
                    GraphTransitionDiagnostic(
                        node_id=target_id,
                        track_id=track_id,
                        candidate_id=None,
                        cost=None,
                        accepted=False,
                        reason="unreachable",
                        incoming_sources=(),
                    )
                )
            continue

        reachable[target_position] = True
        for track_id in range(n_tracks):
            inherited = [
                anchor_memory[positions[edge.source], track_id]
                for edge in target_edges
                if np.isfinite(anchor_memory[positions[edge.source], track_id])
            ]
            if inherited:
                anchor_memory[target_position, track_id] = float(np.mean(inherited))

        if not point.n_candidates:
            for track_id in range(n_tracks):
                diagnostics.append(
                    GraphTransitionDiagnostic(
                        node_id=target_id,
                        track_id=track_id,
                        candidate_id=None,
                        cost=None,
                        accepted=False,
                        reason=("unassigned" if active_tracks[track_id] else "inactive_anchor"),
                        incoming_sources=incoming_sources,
                    )
                )
            continue

        total_costs = np.zeros((n_tracks, point.n_candidates), dtype=float)
        outside_any = np.zeros_like(total_costs, dtype=bool)
        finite_candidates = np.all(np.isfinite(point.features), axis=1)
        proposals_by_track: dict[
            int,
            list[tuple[Hashable, Hashable | None]],
        ] = defaultdict(list)

        for edge in target_edges:
            source_position = positions[edge.source]
            previous_edge = _edge_previous(edge, incoming)
            if previous_edge is None:
                previous = np.full_like(anchor_features, np.nan)
                previous_delta = None
            else:
                previous = features_out[positions[previous_edge.source]]
                previous_delta = previous_edge.step

            edge_costs, edge_finite = transition_costs(
                point.features,
                features_out[source_position],
                previous,
                delta=edge.step,
                previous_delta=previous_delta,
                value_weights=value_weights * edge.value_scale,
                derivative_weights=(
                    derivative_weights * edge.derivative_scale
                ),
                nan_cost_penalty=config.nan_cost_penalty,
            )
            edge_outside = np.zeros_like(edge_costs, dtype=bool)
            if config.anchor_window is not None:
                candidate_values = np.real(
                    point.features[:, config.anchor_feature]
                )
                for track_id in range(n_tracks):
                    anchor_value = anchor_memory[source_position, track_id]
                    if not np.isfinite(anchor_value):
                        continue
                    outside = (
                        np.abs(candidate_values - anchor_value)
                        > config.anchor_window
                    )
                    outside |= ~np.isfinite(candidate_values)
                    edge_outside[track_id] = outside
                    edge_costs[track_id, outside] += _INF_COST

            total_costs += edge_costs
            outside_any |= edge_outside
            informative_tracks = np.flatnonzero(
                active_tracks
                & np.all(np.isfinite(features_out[source_position]), axis=1)
            )
            proposals = _accepted_edge_proposals(
                point,
                edge_costs,
                edge_finite,
                edge_outside,
                active_tracks,
                informative_tracks,
            )
            for track_id, candidate_id in proposals.items():
                proposals_by_track[track_id].append((edge.source, candidate_id))

        if config.rank_penalty > 0:
            candidate_values = np.real(point.features[:, config.anchor_feature])
            ranks = np.argsort(np.argsort(candidate_values))
            for track_id in range(n_tracks):
                total_costs[track_id] += (
                    (ranks - track_id) ** 2 * config.rank_penalty
                )

        assigned_tracks: set[int] = set()
        selected_by_track: dict[int, Hashable | None] = {}
        active_indices = np.flatnonzero(active_tracks)
        rows, columns = linear_sum_assignment(total_costs[active_indices])
        for active_row, candidate_index in zip(rows.tolist(), columns.tolist()):
            track_id = int(active_indices[active_row])
            assigned_tracks.add(track_id)
            candidate_id = point.ids[candidate_index]
            cost = float(total_costs[track_id, candidate_index])
            reason = None
            if not finite_candidates[candidate_index]:
                reason = "non_finite_candidate"
            elif outside_any[track_id, candidate_index]:
                reason = "anchor_window"
            accepted = reason is None
            selected_by_track[track_id] = candidate_id if accepted else None
            if accepted:
                ids_out[target_position][track_id] = candidate_id
                features_out[target_position, track_id] = point.features[candidate_index]
                costs_out[target_position, track_id] = cost
                anchor_value = np.real(
                    point.features[candidate_index, config.anchor_feature]
                )
                if np.isfinite(anchor_value):
                    anchor_memory[target_position, track_id] = anchor_value
            diagnostics.append(
                GraphTransitionDiagnostic(
                    node_id=target_id,
                    track_id=track_id,
                    candidate_id=candidate_id,
                    cost=cost,
                    accepted=accepted,
                    reason=reason,
                    incoming_sources=incoming_sources,
                )
            )

        for track_id in range(n_tracks):
            if track_id in assigned_tracks:
                continue
            selected_by_track[track_id] = None
            diagnostics.append(
                GraphTransitionDiagnostic(
                    node_id=target_id,
                    track_id=track_id,
                    candidate_id=None,
                    cost=None,
                    accepted=False,
                    reason=("unassigned" if active_tracks[track_id] else "inactive_anchor"),
                    incoming_sources=incoming_sources,
                )
            )

        for track_id, proposals in proposals_by_track.items():
            if len(proposals) < 2:
                continue
            if len({candidate_id for _, candidate_id in proposals}) > 1:
                conflicts.append(
                    GraphConflictDiagnostic(
                        node_id=target_id,
                        track_id=track_id,
                        proposals=tuple(proposals),
                        selected_candidate_id=selected_by_track.get(track_id),
                    )
                )

    return GraphTrackingResult(
        anchor_id=anchor_id,
        node_ids=node_ids,
        candidate_ids=tuple(tuple(row) for row in ids_out),
        features=features_out,
        costs=costs_out,
        diagnostics=tuple(diagnostics),
        conflicts=tuple(conflicts),
        unreachable_nodes=tuple(unreachable_nodes),
    )
