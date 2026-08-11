"""Candidate-based mode tracking."""

from .models import (
    CandidateSet,
    GraphConflictDiagnostic,
    GraphTrackingResult,
    GraphTransitionDiagnostic,
    TrackingConfig,
    TrackingResult,
    TransitionDiagnostic,
    TraversalEdge,
)
from .graph import rectilinear_traversal, track_graph
from .path import track_path
from .postprocess import SplitGridResult, split_disconnected_grid_tracks
from .additional import (
    infer_candidate_selector,
    pick_candidate_data,
    pick_candidate_data_stable,
)
from .legacy_hungarian import (
    estimate_derivative_one_sided_vector_component,
    group_vectors_one_sided_hungarian,
    ordered_vectors_by_abs,
)

__all__ = [
    "CandidateSet",
    "GraphConflictDiagnostic",
    "GraphTrackingResult",
    "GraphTransitionDiagnostic",
    "TrackingConfig",
    "TrackingResult",
    "SplitGridResult",
    "TransitionDiagnostic",
    "TraversalEdge",
    "rectilinear_traversal",
    "track_graph",
    "track_path",
    "split_disconnected_grid_tracks",
    "infer_candidate_selector",
    "pick_candidate_data",
    "pick_candidate_data_stable",
    "estimate_derivative_one_sided_vector_component",
    "group_vectors_one_sided_hungarian",
    "ordered_vectors_by_abs",
]
