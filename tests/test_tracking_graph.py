from __future__ import annotations

import numpy as np
import pytest

from eigenmode_analysis.tracking import (
    CandidateSet,
    TrackingConfig,
    TraversalEdge,
    rectilinear_traversal,
    track_graph,
    track_path,
)


def test_graph_path_reduction_matches_one_dimensional_tracker():
    coordinates = np.linspace(0.0, 1.0, 31)
    values = np.stack(
        [
            1.0 + 0.8 * coordinates,
            1.8 - 0.8 * coordinates,
        ],
        axis=1,
    )
    points = {}
    path_points = []
    for index, row in enumerate(values):
        order = np.array([1, 0]) if index % 2 else np.array([0, 1])
        point = CandidateSet(order.tolist(), row[order, None])
        points[index] = point
        path_points.append(point)
    edges = tuple(
        TraversalEdge(
            index - 1,
            index,
            axis=0,
            step=float(coordinates[index] - coordinates[index - 1]),
        )
        for index in range(1, len(coordinates))
    )
    config = TrackingConfig(n_tracks=2, derivative_weights=2.0)

    graph_result = track_graph(
        points,
        node_order=tuple(range(len(points))),
        edges=edges,
        anchor_id=0,
        config=config,
    )
    path_result = track_path(
        path_points,
        coordinates=coordinates,
        config=config,
    )

    assert graph_result.candidate_ids == path_result.candidate_ids
    np.testing.assert_allclose(graph_result.features, path_result.features)
    assert graph_result.conflicts == ()


def _smooth_grid_points(shape, *, reverse_candidates=False):
    points = {}
    for index in np.ndindex(*shape):
        if index == (2, 2):
            points[index] = CandidateSet([], np.empty((0, 1)))
            continue
        i, j = index
        ids = ["low", "high"]
        features = np.array([[1.0 + 0.1 * i + 0.05 * j], [2.0 - 0.05 * i + 0.08 * j]])
        if reverse_candidates ^ ((i + j) % 2 == 1):
            ids.reverse()
            features = features[::-1]
        points[index] = CandidateSet(ids, features)
    return points


def test_rectilinear_graph_routes_around_hole_and_is_input_order_invariant():
    shape = (5, 5)
    node_order, edges = rectilinear_traversal(shape, (0.1, 0.2))
    config = TrackingConfig(
        n_tracks=2,
        derivative_weights=0.2,
        anchor_window=0.5,
    )

    forward = track_graph(
        _smooth_grid_points(shape),
        node_order=node_order,
        edges=edges,
        anchor_id=(0, 0),
        config=config,
    )
    permuted = track_graph(
        _smooth_grid_points(shape, reverse_candidates=True),
        node_order=node_order,
        edges=tuple(reversed(edges)),
        anchor_id=(0, 0),
        config=config,
    )

    assert forward.candidate_ids == permuted.candidate_ids
    assert forward.candidate_ids_at((2, 2)) == (None, None)
    assert forward.candidate_ids_at((4, 4)) == ("low", "high")
    assert forward.unreachable_nodes == ()
    assert forward.conflicts == ()


def test_graph_rejects_non_finite_candidate_without_corrupting_assignment():
    points = {
        "anchor": CandidateSet(["a-low", "a-high"], [[1.0], [2.0]]),
        "next": CandidateSet(
            ["bad", "high", "low"],
            [[np.nan], [2.1], [1.1]],
        ),
    }

    result = track_graph(
        points,
        node_order=("anchor", "next"),
        edges=(TraversalEdge("anchor", "next"),),
        anchor_id="anchor",
        config=TrackingConfig(n_tracks=2),
    )

    assert result.candidate_ids_at("next") == ("low", "high")
    assert all(
        diagnostic.candidate_id != "bad"
        for diagnostic in result.diagnostics
        if diagnostic.accepted
    )


def test_graph_window_recovers_across_empty_reachable_node():
    points = {
        "anchor": CandidateSet(["a-low", "a-high"], [[1.0], [2.0]]),
        "empty": CandidateSet([], np.empty((0, 1))),
        "resume": CandidateSet(
            ["distractor", "high", "low"],
            [[5.0], [2.05], [1.05]],
        ),
    }

    result = track_graph(
        points,
        node_order=("anchor", "empty", "resume"),
        edges=(
            TraversalEdge("anchor", "empty", axis="path"),
            TraversalEdge("empty", "resume", axis="path"),
        ),
        anchor_id="anchor",
        config=TrackingConfig(n_tracks=2, anchor_window=0.2),
    )

    assert result.candidate_ids_at("empty") == (None, None)
    assert result.candidate_ids_at("resume") == ("low", "high")
    assert result.unreachable_nodes == ()


def test_merge_disagreement_is_returned_as_conflict_diagnostic():
    points = {
        "anchor": CandidateSet(["a0", "a1"], [[0.0], [10.0]]),
        "b1": CandidateSet(["b10", "b11"], [[4.0], [6.0]]),
        "c1": CandidateSet(["c10", "c11"], [[0.0], [10.0]]),
        "b2": CandidateSet(["b20", "b21"], [[8.0], [2.0]]),
        "c2": CandidateSet(["c20", "c21"], [[0.0], [10.0]]),
        "merge": CandidateSet(["low", "high"], [[1.0], [9.0]]),
    }
    node_order = ("anchor", "b1", "c1", "b2", "c2", "merge")
    edges = (
        TraversalEdge("anchor", "b1", axis="b"),
        TraversalEdge("anchor", "c1", axis="c"),
        TraversalEdge("b1", "b2", axis="b"),
        TraversalEdge("c1", "c2", axis="c"),
        TraversalEdge("b2", "merge", axis="b"),
        TraversalEdge("c2", "merge", axis="c"),
    )

    result = track_graph(
        points,
        node_order=node_order,
        edges=edges,
        anchor_id="anchor",
        config=TrackingConfig(n_tracks=2, derivative_weights=2.0),
    )

    merge_conflicts = [
        conflict for conflict in result.conflicts if conflict.node_id == "merge"
    ]
    assert {conflict.track_id for conflict in merge_conflicts} == {0, 1}
    assert {
        candidate_id
        for _, candidate_id in merge_conflicts[0].proposals
    } == {"low", "high"}


def test_disconnected_nodes_are_unreachable_not_silently_initialized():
    points = {
        "anchor": CandidateSet([0, 1], [[1.0], [2.0]]),
        "reachable": CandidateSet([10, 11], [[1.1], [2.1]]),
        "isolated": CandidateSet([20, 21], [[1.2], [2.2]]),
    }

    result = track_graph(
        points,
        node_order=("anchor", "reachable", "isolated"),
        edges=(TraversalEdge("anchor", "reachable"),),
        anchor_id="anchor",
        config=TrackingConfig(n_tracks=2),
    )

    assert result.unreachable_nodes == ("isolated",)
    assert result.candidate_ids_at("isolated") == (None, None)
    assert [
        diagnostic.reason
        for diagnostic in result.diagnostics
        if diagnostic.node_id == "isolated"
    ] == ["unreachable", "unreachable"]


def test_graph_does_not_create_track_absent_at_anchor():
    points = {
        0: CandidateSet(["existing"], [[1.0]]),
        1: CandidateSet(["existing", "born"], [[1.1], [2.0]]),
    }

    result = track_graph(
        points,
        node_order=(0, 1),
        edges=(TraversalEdge(0, 1),),
        anchor_id=0,
        config=TrackingConfig(n_tracks=2),
    )

    assert result.candidate_ids == (("existing", None), ("existing", None))
    assert result.diagnostics[-1].reason == "inactive_anchor"


def test_graph_requires_explicit_forward_topology():
    points = {
        "anchor": CandidateSet([0], [[1.0]]),
        "next": CandidateSet([1], [[1.1]]),
    }

    with pytest.raises(ValueError, match="first node"):
        track_graph(
            points,
            node_order=("next", "anchor"),
            edges=(),
            anchor_id="anchor",
        )
    with pytest.raises(ValueError, match="point forward"):
        track_graph(
            points,
            node_order=("anchor", "next"),
            edges=(TraversalEdge("next", "anchor"),),
            anchor_id="anchor",
        )


def test_graph_rejects_undefined_multi_path_slope_policy():
    points = {"anchor": CandidateSet([0], [[1.0]])}

    with pytest.raises(NotImplementedError, match="path-distance policy"):
        track_graph(
            points,
            node_order=("anchor",),
            edges=(),
            anchor_id="anchor",
            config=TrackingConfig(anchor_slope=1.0),
        )
