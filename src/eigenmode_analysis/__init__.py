"""Schema-neutral numerical primitives for eigenmode research."""

from ._version import __version__
from .tracking import (
    CandidateSet,
    GraphConflictDiagnostic,
    GraphTrackingResult,
    GraphTransitionDiagnostic,
    TrackingConfig,
    TrackingResult,
    SplitGridResult,
    TransitionDiagnostic,
    TraversalEdge,
    rectilinear_traversal,
    track_graph,
    track_path,
    split_disconnected_grid_tracks,
)
from .symmetry import (
    complete_C2_polarization,
    complete_C4_polarization,
    complete_C4_spectrum,
    complete_C6_polarization,
    geom_complete,
)
from .topology import integrate_skyrmion_density, skyrmion_density
from .polarization import (
    ellipse_to_stokes,
    jones_from_stokes,
    normalize_stokes,
    poincare_points,
    stokes_from_jones,
    stokes_to_ellipse,
)
from .peak_tracking import (
    find_peaks_on_1d_metric,
    hungarian_track_by_distance,
    metric_Q_from_complex,
    pack_peaks_records,
)
from .grid_sampling import (
    bilinear_interpolate,
    bilinear_interpolate_angle_pi,
    find_cell_edges,
    resample_path_uniform,
    sample_fields_along_path,
    wrap_angle_pi,
)
from .field_regimes import extract_field_splits
from .lineshapes import (
    FitResult as LineshapeFitResult,
    fano,
    fit_both_and_compare,
    fit_lineshape,
    lorentzian,
)
from .propagation import angular_spectrum_propagate, angular_spectrum_xz_slice

__all__ = [
    "__version__",
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
    "geom_complete",
    "complete_C2_polarization",
    "complete_C4_polarization",
    "complete_C6_polarization",
    "complete_C4_spectrum",
    "ellipse_to_stokes",
    "stokes_to_ellipse",
    "skyrmion_density",
    "integrate_skyrmion_density",
    "stokes_from_jones",
    "jones_from_stokes",
    "normalize_stokes",
    "poincare_points",
    "metric_Q_from_complex",
    "find_peaks_on_1d_metric",
    "pack_peaks_records",
    "hungarian_track_by_distance",
    "wrap_angle_pi",
    "find_cell_edges",
    "bilinear_interpolate",
    "bilinear_interpolate_angle_pi",
    "resample_path_uniform",
    "sample_fields_along_path",
    "LineshapeFitResult",
    "lorentzian",
    "fano",
    "fit_lineshape",
    "fit_both_and_compare",
    "angular_spectrum_propagate",
    "angular_spectrum_xz_slice",
    "extract_field_splits",
]
