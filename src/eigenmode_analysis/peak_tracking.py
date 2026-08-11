"""Peak metrics and distance-based tracking primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


def metric_Q_from_complex(values: np.ndarray) -> np.ndarray:
    """Compute the signed ``Re(z) / (2 Im(z))`` metric."""

    array = np.asarray(values, dtype=complex)
    real = np.real(array)
    imag = np.imag(array)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(imag != 0, real / (2 * imag), 1e15)


def find_peaks_on_1d_metric(metric_1d: np.ndarray, *, height: float):
    """Run SciPy peak detection after applying the documented NaN policy."""

    values = np.nan_to_num(metric_1d, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    peaks, properties = find_peaks(values, height=height)
    return np.asarray(peaks), properties


def pack_peaks_records(
    peaks_idx: np.ndarray,
    *,
    coords_1d: Mapping[str, Sequence[float]],
    Z_target_1d: np.ndarray,
    metric_1d: np.ndarray,
    k_key: str = "k",
) -> list[dict[str, Any]]:
    """Pack peak indices into coordinate/frequency records."""

    values = np.asarray(Z_target_1d, dtype=complex)
    found = []
    for peak in peaks_idx:
        k_value = coords_1d[k_key][peak]
        frequency = values[peak] if peak < len(values) else np.nan
        metric = metric_1d[peak] if peak < len(metric_1d) else np.nan
        found.append(
            {
                "k": float(k_value),
                "freq": complex(frequency),
                "metric": float(metric),
                "index": int(peak),
            }
        )
    return found


def hungarian_track_by_distance(
    scan_values_sorted: np.ndarray,
    peaks_by_scan: Mapping[float, Sequence[Mapping[str, Any]]],
    *,
    get_point_coord: Callable[[Mapping[str, Any]], float] = lambda peak: peak["k"],
    max_distance: float = 0.1,
    fill_value=np.nan,
) -> list[dict[str, list[Any]]]:
    """Track peak records between scans using gated Hungarian distance."""

    tracks = []
    if len(scan_values_sorted) == 0:
        return tracks

    first = float(scan_values_sorted[0])
    for peak in peaks_by_scan.get(first, []):
        tracks.append(
            {
                "buffers": [first],
                "ks": [peak.get("k", fill_value)],
                "freqs": [peak.get("freq", fill_value)],
                "metrics": [peak.get("metric", fill_value)],
            }
        )

    for index in range(1, len(scan_values_sorted)):
        current = float(scan_values_sorted[index])
        current_peaks = peaks_by_scan.get(current, [])

        if len(current_peaks) == 0:
            for track in tracks:
                track["buffers"].append(current)
                track["ks"].append(fill_value)
                track["freqs"].append(fill_value)
                track["metrics"].append(fill_value)
            continue

        if len(tracks) == 0:
            for peak in current_peaks:
                tracks.append(
                    {
                        "buffers": [current],
                        "ks": [peak.get("k", fill_value)],
                        "freqs": [peak.get("freq", fill_value)],
                        "metrics": [peak.get("metric", fill_value)],
                    }
                )
            continue

        reference_coordinates = []
        for track in tracks:
            values = np.asarray(track["ks"], dtype=float)
            reference = np.nan if np.all(np.isnan(values)) else float(values[~np.isnan(values)][-1])
            reference_coordinates.append(reference)

        current_coordinates = np.asarray(
            [get_point_coord(peak) for peak in current_peaks], dtype=float
        )
        distances = np.full(
            (len(reference_coordinates), len(current_coordinates)), 1e6, dtype=float
        )
        for row, reference in enumerate(reference_coordinates):
            if not np.isnan(reference):
                distances[row, :] = np.abs(current_coordinates - reference)

        rows, columns = linear_sum_assignment(distances)
        assigned_columns = set()
        for row, column in zip(rows, columns):
            track = tracks[row]
            if distances[row, column] <= max_distance:
                peak = current_peaks[column]
                track["buffers"].append(current)
                track["ks"].append(peak.get("k", fill_value))
                track["freqs"].append(peak.get("freq", fill_value))
                track["metrics"].append(peak.get("metric", fill_value))
                assigned_columns.add(column)
            else:
                track["buffers"].append(current)
                track["ks"].append(fill_value)
                track["freqs"].append(fill_value)
                track["metrics"].append(fill_value)

        for column, peak in enumerate(current_peaks):
            if column in assigned_columns:
                continue
            track = {"buffers": [], "ks": [], "freqs": [], "metrics": []}
            for previous in range(index):
                track["buffers"].append(float(scan_values_sorted[previous]))
                track["ks"].append(fill_value)
                track["freqs"].append(fill_value)
                track["metrics"].append(fill_value)
            track["buffers"].append(current)
            track["ks"].append(peak.get("k", fill_value))
            track["freqs"].append(peak.get("freq", fill_value))
            track["metrics"].append(peak.get("metric", fill_value))
            tracks.append(track)

    expected = len(scan_values_sorted)
    for track in tracks:
        while len(track["buffers"]) < expected:
            track["buffers"].append(float(scan_values_sorted[len(track["buffers"])]))
            track["ks"].append(fill_value)
            track["freqs"].append(fill_value)
            track["metrics"].append(fill_value)
    return tracks


__all__ = [
    "metric_Q_from_complex",
    "find_peaks_on_1d_metric",
    "pack_peaks_records",
    "hungarian_track_by_distance",
]
