from __future__ import annotations

import numpy as np
import pytest

from eigenmode_analysis.tracking import CandidateSet, TrackingConfig, track_path


def _shuffled_points(values: np.ndarray, *, seed: int = 20260805):
    rng = np.random.default_rng(seed)
    points = []
    for row in values:
        order = rng.permutation(values.shape[1])
        points.append(CandidateSet(order.tolist(), row[order, None]))
    return points


def test_derivative_continuity_tracks_a_true_crossing():
    x = np.linspace(0.0, 1.0, 61)
    values = np.stack(
        [
            1.0 + 0.8 * x + 0.006j,
            1.8 - 0.8 * x + 0.009j,
        ],
        axis=1,
    )
    result = track_path(
        _shuffled_points(values),
        coordinates=x,
        config=TrackingConfig(n_tracks=2, derivative_weights=2.0),
    )

    assert all(row == (0, 1) for row in result.candidate_ids)
    np.testing.assert_allclose(result.features[:, :, 0], values)


def test_exactly_identical_features_do_not_claim_hidden_identity():
    points = [
        CandidateSet(["a", "b"], [[1.0], [2.0]]),
        CandidateSet(["b", "a"], [[1.5], [1.5]]),
        CandidateSet(["a", "b"], [[2.0], [1.0]]),
    ]
    result = track_path(
        points,
        coordinates=[0.0, 1.0, 2.0],
        config=TrackingConfig(n_tracks=2, derivative_weights=2.0),
    )

    assert set(result.candidate_ids[1]) == {"a", "b"}
    assert np.all(result.features[1, :, 0] == 1.5)


def test_anchor_window_preserves_identity_across_missing_segment():
    x = np.arange(7, dtype=float)
    points = [
        CandidateSet(["low", "high"], [[1.00 + 0.01 * i], [1.50 + 0.01 * i]])
        for i in range(len(x))
    ]
    points[3] = CandidateSet(["high"], [[1.53]])
    points[4] = CandidateSet(["high"], [[1.54]])

    result = track_path(
        points,
        coordinates=x,
        config=TrackingConfig(n_tracks=2, anchor_window=0.08),
    )

    assert result.candidate_ids[3][0] is None
    assert result.candidate_ids[4][0] is None
    assert result.candidate_ids[5][0] == "low"
    assert result.candidate_ids[6][0] == "low"


@pytest.mark.parametrize(
    "coordinates",
    [
        np.linspace(0.0, 1.0, 11),
        np.linspace(0.0, 1.0, 3),
    ],
)
def test_anchor_slope_is_independent_of_path_sampling_resolution(coordinates):
    points = [
        CandidateSet(
            ["mode", "distractor"],
            [[100.0 + 10.0 * coordinate], [140.0 - 2.0 * coordinate]],
        )
        for coordinate in coordinates
    ]

    result = track_path(
        points,
        coordinates=coordinates,
        config=TrackingConfig(n_tracks=1, anchor_slope=12.0),
    )

    assert all(row == ("mode",) for row in result.candidate_ids)
    assert result.coverage == 1.0


def test_anchor_slope_expands_from_last_accepted_point_across_missing_segment():
    coordinates = np.arange(5, dtype=float)
    points = [
        CandidateSet(["mode", "other"], [[1.0], [2.0]]),
        CandidateSet(["mode", "other"], [[1.1], [2.0]]),
        CandidateSet(["other"], [[2.0]]),
        CandidateSet(["other"], [[2.0]]),
        CandidateSet(["mode", "other"], [[1.4], [2.0]]),
    ]

    result = track_path(
        points,
        coordinates=coordinates,
        config=TrackingConfig(n_tracks=1, anchor_slope=0.15),
    )

    assert result.candidate_ids[2][0] is None
    assert result.candidate_ids[3][0] is None
    assert result.candidate_ids[4][0] == "mode"
    assert [
        diagnostic.reason
        for diagnostic in result.diagnostics
        if diagnostic.point_index in {2, 3}
    ] == ["anchor_slope", "anchor_slope"]


def test_anchor_window_and_slope_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrackingConfig(anchor_window=1.0, anchor_slope=1.0)


def test_track_missing_at_anchor_does_not_silently_claim_later_mode_birth():
    points = [
        CandidateSet(["existing"], [[1.0]]),
        CandidateSet(["existing", "born"], [[1.1], [2.0]]),
        CandidateSet(["existing", "born"], [[1.2], [2.1]]),
    ]

    result = track_path(points, config=TrackingConfig(n_tracks=2))

    assert result.candidate_ids == (
        ("existing", None),
        ("existing", None),
        ("existing", None),
    )
    assert [
        diagnostic.reason
        for diagnostic in result.diagnostics
        if diagnostic.track_id == 1
    ] == ["unassigned", "unassigned"]


def test_non_finite_candidate_is_rejected():
    points = [
        CandidateSet([0, 1], [[1.0], [2.0]]),
        CandidateSet([10, 11], [[np.nan], [2.1]]),
    ]
    result = track_path(points, config=TrackingConfig(n_tracks=2))

    assert sum(cid is not None for cid in result.candidate_ids[1]) == 1
    assert any(d.reason == "non_finite_candidate" for d in result.diagnostics)


def test_middle_anchor_seeds_both_directions_consistently():
    coordinates = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    values = np.stack(
        [
            1.0 + 0.05 * coordinates,
            1.5 - 0.03 * coordinates,
        ],
        axis=1,
    )
    points = _shuffled_points(values)
    result = track_path(
        points,
        coordinates=coordinates,
        anchor_index=2,
        config=TrackingConfig(n_tracks=2),
    )

    assert result.anchor_index == 2
    assert all(row == (0, 1) for row in result.candidate_ids)


def test_candidate_set_validates_identity_contract():
    with pytest.raises(ValueError, match="unique"):
        CandidateSet([1, 1], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="same number"):
        CandidateSet([1], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="numeric"):
        CandidateSet([1], [["not-a-number"]])


def test_rank_penalty_prefers_matching_frequency_rank():
    points = [
        CandidateSet([0, 1], [[1.0], [2.0]]),
        CandidateSet([10, 11, 12], [[1.9], [1.1], [3.0]]),
    ]
    result = track_path(
        points,
        config=TrackingConfig(n_tracks=2, rank_penalty=100.0),
    )

    assert result.candidate_ids[1] == (11, 10)
