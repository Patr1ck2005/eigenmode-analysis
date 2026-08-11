from __future__ import annotations

import numpy as np

from eigenmode_analysis import (
    find_peaks_on_1d_metric,
    hungarian_track_by_distance,
    metric_Q_from_complex,
    pack_peaks_records,
)


def test_metric_and_peak_record_contract():
    frequencies = np.array([1 + 0.5j, 2 + 0j, 3 + 0.5j, 4 + 1j, 5 + 0.5j])
    metric = metric_Q_from_complex(frequencies)
    np.testing.assert_allclose(metric, [1.0, 1e15, 3.0, 2.0, 5.0])

    peaks, properties = find_peaks_on_1d_metric(metric, height=2.5)
    np.testing.assert_array_equal(peaks, [1])
    assert properties["peak_heights"][0] == 1e15

    records = pack_peaks_records(
        peaks,
        coords_1d={"k": np.linspace(-1, 1, len(metric))},
        Z_target_1d=frequencies,
        metric_1d=metric,
    )
    assert records == [{"k": -0.5, "freq": 2 + 0j, "metric": 1e15, "index": 1}]


def test_distance_tracking_preserves_missing_scans_and_adds_new_tracks():
    scans = np.array([0.0, 1.0, 2.0, 3.0])
    peaks = {
        0.0: [{"k": 0.0, "freq": 1 + 0j, "metric": 10.0}],
        1.0: [{"k": 0.04, "freq": 1.1 + 0j, "metric": 11.0}],
        2.0: [],
        3.0: [
            {"k": 0.08, "freq": 1.2 + 0j, "metric": 12.0},
            {"k": 0.8, "freq": 2 + 0j, "metric": 20.0},
        ],
    }

    tracks = hungarian_track_by_distance(scans, peaks, max_distance=0.1)

    assert len(tracks) == 2
    np.testing.assert_allclose(tracks[0]["ks"], [0.0, 0.04, np.nan, 0.08], equal_nan=True)
    np.testing.assert_allclose(tracks[1]["ks"], [np.nan, np.nan, np.nan, 0.8], equal_nan=True)
    assert all(len(track["buffers"]) == len(scans) for track in tracks)


def test_distance_gate_rejects_jump_and_starts_new_track():
    tracks = hungarian_track_by_distance(
        np.array([0.0, 1.0]),
        {
            0.0: [{"k": 0.0, "freq": 1.0, "metric": 1.0}],
            1.0: [{"k": 0.5, "freq": 2.0, "metric": 2.0}],
        },
        max_distance=0.1,
    )
    assert len(tracks) == 2
    np.testing.assert_allclose(tracks[0]["ks"], [0.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(tracks[1]["ks"], [np.nan, 0.5], equal_nan=True)
