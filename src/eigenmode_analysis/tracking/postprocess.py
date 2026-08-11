"""Schema-neutral postprocessing for tracked grid streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
from scipy.stats import median_abs_deviation


@dataclass(frozen=True)
class SplitGridResult:
    """Track arrays after discontinuous connected components are separated."""

    node_ids: tuple[tuple[int, ...], ...]
    candidate_ids: tuple[tuple[Hashable | None, ...], ...]
    features: np.ndarray
    moved_segments: tuple[tuple[int, int, int], ...]
    discarded_segments: tuple[tuple[int, int], ...]


def split_disconnected_grid_tracks(
    node_ids: Sequence[Sequence[int]],
    candidate_ids: Sequence[Sequence[Hashable | None]],
    features,
    *,
    shape: Sequence[int],
    min_segment_points: int = 3,
    mad_multiplier: float = 5.0,
) -> SplitGridResult:
    """Separate jump-disconnected components into currently empty tracks.

    Connectivity is defined on direct grid
    neighbours. An edge is retained when its feature jump is no larger than
    ``median + mad_multiplier * normalized_MAD`` for that source track.
    """

    shape = tuple(int(size) for size in shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("shape must contain positive axis sizes")
    if min_segment_points <= 0:
        raise ValueError("min_segment_points must be positive")
    if mad_multiplier < 0 or not np.isfinite(mad_multiplier):
        raise ValueError("mad_multiplier must be finite and non-negative")

    nodes = tuple(tuple(int(value) for value in node) for node in node_ids)
    if len(nodes) != int(np.prod(shape)) or set(nodes) != set(np.ndindex(*shape)):
        raise ValueError("node_ids must contain every grid coordinate exactly once")
    positions = {node: position for position, node in enumerate(nodes)}

    feature_array = np.asarray(features)
    if feature_array.ndim != 3 or feature_array.shape[0] != len(nodes):
        raise ValueError("features must have shape (n_nodes, n_tracks, n_features)")
    ids = [list(row) for row in candidate_ids]
    n_tracks = feature_array.shape[1]
    if len(ids) != len(nodes) or any(len(row) != n_tracks for row in ids):
        raise ValueError("candidate_ids must have shape (n_nodes, n_tracks)")

    output_features = feature_array.copy()
    output_ids = [row.copy() for row in ids]
    moved: list[tuple[int, int, int]] = []
    discarded: list[tuple[int, int]] = []

    def finite_track(position: int, track: int) -> bool:
        return bool(np.all(np.isfinite(output_features[position, track])))

    def empty_track(track: int) -> bool:
        return not any(finite_track(position, track) for position in range(len(nodes)))

    for source_track in range(n_tracks):
        populated = [
            position
            for position in range(len(nodes))
            if finite_track(position, source_track)
        ]
        if len(populated) < 3:
            continue
        populated_set = set(populated)
        edges: list[tuple[int, int, float]] = []
        jumps: list[float] = []
        for position in populated:
            node = nodes[position]
            for axis in range(len(shape)):
                neighbour = list(node)
                neighbour[axis] += 1
                if neighbour[axis] >= shape[axis]:
                    continue
                other = positions[tuple(neighbour)]
                if other not in populated_set:
                    continue
                jump = float(
                    np.linalg.norm(
                        output_features[position, source_track]
                        - output_features[other, source_track]
                    )
                )
                edges.append((position, other, jump))
                jumps.append(jump)
        if not jumps:
            continue
        median = float(np.median(jumps))
        mad = float(median_abs_deviation(jumps, scale="normal"))
        threshold = median + mad_multiplier * max(mad, 1e-12)
        adjacency = {position: [] for position in populated}
        for first, second, jump in edges:
            if jump <= threshold:
                adjacency[first].append(second)
                adjacency[second].append(first)

        components: list[list[int]] = []
        visited: set[int] = set()
        for start in populated:
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            component = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbour in adjacency[current]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        stack.append(neighbour)
            if len(component) >= min_segment_points:
                components.append(component)
        components.sort(key=len, reverse=True)
        if len(components) <= 1:
            continue

        saved = [
            (
                component,
                output_features[component, source_track].copy(),
                [output_ids[position][source_track] for position in component],
            )
            for component in components
        ]
        for position in populated:
            output_features[position, source_track] = np.nan
            output_ids[position][source_track] = None

        target = source_track
        for component, component_features, component_ids in saved:
            while target < n_tracks and not empty_track(target):
                target += 1
            if target >= n_tracks:
                discarded.append((source_track, len(component)))
                continue
            for offset, position in enumerate(component):
                output_features[position, target] = component_features[offset]
                output_ids[position][target] = component_ids[offset]
            moved.append((source_track, target, len(component)))
            target += 1

    return SplitGridResult(
        node_ids=nodes,
        candidate_ids=tuple(tuple(row) for row in output_ids),
        features=output_features,
        moved_segments=tuple(moved),
        discarded_segments=tuple(discarded),
    )
