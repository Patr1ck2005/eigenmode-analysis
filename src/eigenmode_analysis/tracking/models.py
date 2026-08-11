"""Public data contracts for candidate-based tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateSet:
    """Candidates available at one point of a scan path.

    ``ids`` are opaque identifiers owned by the caller. ``features`` has shape
    ``(n_candidates, n_features)`` and may be real or complex. Non-finite
    features are retained so the tracker can reject them explicitly.
    """

    ids: tuple[Hashable, ...]
    features: np.ndarray

    def __init__(self, ids: Sequence[Hashable], features) -> None:
        ids_tuple = tuple(ids)
        feature_array = np.asarray(features)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(-1, 1)
        if feature_array.ndim != 2:
            raise ValueError("features must have shape (n_candidates, n_features)")
        if not np.issubdtype(feature_array.dtype, np.number):
            raise ValueError("features must contain numeric values")
        if feature_array.shape[0] != len(ids_tuple):
            raise ValueError("ids and features must contain the same number of candidates")
        if len(set(ids_tuple)) != len(ids_tuple):
            raise ValueError("candidate ids must be unique within each path point")
        object.__setattr__(self, "ids", ids_tuple)
        object.__setattr__(self, "features", feature_array.copy())

    @property
    def n_candidates(self) -> int:
        return len(self.ids)

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True)
class TrackingConfig:
    """Numerical policy for path tracking."""

    n_tracks: int | None = None
    value_weights: float | Sequence[float] = 1.0
    derivative_weights: float | Sequence[float] = 0.1
    nan_cost_penalty: float = 10.0
    anchor_feature: int = 0
    anchor_window: float | None = None
    anchor_slope: float | None = None
    rank_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.n_tracks is not None and self.n_tracks <= 0:
            raise ValueError("n_tracks must be positive")
        if self.nan_cost_penalty <= 0:
            raise ValueError("nan_cost_penalty must be positive")
        if self.anchor_feature < 0:
            raise ValueError("anchor_feature must be non-negative")
        if self.anchor_window is not None and (
            self.anchor_window <= 0 or not np.isfinite(self.anchor_window)
        ):
            raise ValueError("anchor_window must be finite and positive when provided")
        if self.anchor_slope is not None and (
            self.anchor_slope <= 0 or not np.isfinite(self.anchor_slope)
        ):
            raise ValueError("anchor_slope must be finite and positive when provided")
        if self.anchor_window is not None and self.anchor_slope is not None:
            raise ValueError("anchor_window and anchor_slope are mutually exclusive")
        if self.rank_penalty < 0 or not np.isfinite(self.rank_penalty):
            raise ValueError("rank_penalty must be finite and non-negative")


@dataclass(frozen=True)
class TransitionDiagnostic:
    """Outcome for one track at one non-anchor path point."""

    point_index: int
    previous_point_index: int
    track_id: int
    candidate_id: Hashable | None
    cost: float | None
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class TrackingResult:
    """Tracked candidate identity, features, and diagnostics."""

    anchor_index: int
    coordinates: np.ndarray
    candidate_ids: tuple[tuple[Hashable | None, ...], ...]
    features: np.ndarray
    costs: np.ndarray
    diagnostics: tuple[TransitionDiagnostic, ...]

    @property
    def n_points(self) -> int:
        return len(self.candidate_ids)

    @property
    def n_tracks(self) -> int:
        return len(self.candidate_ids[0]) if self.candidate_ids else 0

    @property
    def coverage(self) -> float:
        if not self.candidate_ids or self.n_tracks == 0:
            return 0.0
        assigned = sum(cid is not None for row in self.candidate_ids for cid in row)
        return assigned / (self.n_points * self.n_tracks)


@dataclass(frozen=True)
class TraversalEdge:
    """One directed edge in an explicit tracking DAG."""

    source: Hashable
    target: Hashable
    axis: Hashable | None = None
    step: float = 1.0
    value_scale: float = 1.0
    derivative_scale: float = 1.0

    def __post_init__(self) -> None:
        hash(self.source)
        hash(self.target)
        if self.axis is not None:
            hash(self.axis)
        if self.source == self.target:
            raise ValueError("traversal edges cannot be self loops")
        if self.step <= 0 or not np.isfinite(self.step):
            raise ValueError("edge step must be finite and positive")
        for name, value in (
            ("value_scale", self.value_scale),
            ("derivative_scale", self.derivative_scale),
        ):
            if value < 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class GraphTransitionDiagnostic:
    """Assignment outcome for one graph node and track."""

    node_id: Hashable
    track_id: int
    candidate_id: Hashable | None
    cost: float | None
    accepted: bool
    reason: str | None
    incoming_sources: tuple[Hashable, ...]


@dataclass(frozen=True)
class GraphConflictDiagnostic:
    """Independent incoming edges proposed different candidate identities."""

    node_id: Hashable
    track_id: int
    proposals: tuple[tuple[Hashable, Hashable | None], ...]
    selected_candidate_id: Hashable | None


@dataclass(frozen=True)
class GraphTrackingResult:
    """Tracked identities and diagnostics in explicit node order."""

    anchor_id: Hashable
    node_ids: tuple[Hashable, ...]
    candidate_ids: tuple[tuple[Hashable | None, ...], ...]
    features: np.ndarray
    costs: np.ndarray
    diagnostics: tuple[GraphTransitionDiagnostic, ...]
    conflicts: tuple[GraphConflictDiagnostic, ...]
    unreachable_nodes: tuple[Hashable, ...]

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_tracks(self) -> int:
        return len(self.candidate_ids[0]) if self.candidate_ids else 0

    @property
    def coverage(self) -> float:
        if not self.candidate_ids or self.n_tracks == 0:
            return 0.0
        assigned = sum(cid is not None for row in self.candidate_ids for cid in row)
        return assigned / (self.n_nodes * self.n_tracks)

    def candidate_ids_at(self, node_id: Hashable) -> tuple[Hashable | None, ...]:
        try:
            position = self.node_ids.index(node_id)
        except ValueError as exc:
            raise KeyError(node_id) from exc
        return self.candidate_ids[position]
