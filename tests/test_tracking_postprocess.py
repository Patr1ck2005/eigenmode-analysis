from __future__ import annotations

import numpy as np

from eigenmode_analysis.tracking import split_disconnected_grid_tracks


def test_auto_split_moves_jump_disconnected_component_to_empty_track():
    shape = (8,)
    node_ids = tuple(np.ndindex(*shape))
    features = np.full((8, 2, 1), np.nan)
    features[:4, 0, 0] = [0.0, 0.1, 0.2, 0.3]
    features[4:, 0, 0] = [10.0, 10.1, 10.2, 10.3]
    candidate_ids = tuple((index, None) for index in range(8))

    result = split_disconnected_grid_tracks(
        node_ids,
        candidate_ids,
        features,
        shape=shape,
        min_segment_points=3,
        mad_multiplier=5.0,
    )

    np.testing.assert_allclose(result.features[:4, 0, 0], [0.0, 0.1, 0.2, 0.3])
    assert np.all(np.isnan(result.features[4:, 0, 0]))
    np.testing.assert_allclose(result.features[4:, 1, 0], [10.0, 10.1, 10.2, 10.3])
    assert result.moved_segments == ((0, 0, 4), (0, 1, 4))


def test_auto_split_preserves_continuous_grid_track():
    shape = (3, 3)
    node_ids = tuple(np.ndindex(*shape))
    features = np.full((9, 2, 1), np.nan)
    for position, (i, j) in enumerate(node_ids):
        features[position, 0, 0] = i + 0.1 * j
    candidate_ids = tuple((position, None) for position in range(9))

    result = split_disconnected_grid_tracks(
        node_ids,
        candidate_ids,
        features,
        shape=shape,
    )

    np.testing.assert_allclose(result.features, features, equal_nan=True)
    assert result.moved_segments == ()
