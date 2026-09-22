"""Typed numerical boundary for the frozen Stable v1 grouping policy."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
import numpy as np

from .._version import __version__
from ._stable.model import CandidateFibers as BandCandidates
from ._stable.model import ParameterCellComplex as ParameterGrid
from ._stable.stable_v1 import Config as GroupingConfig
from ._stable.stable_v1 import VERSION as ALGORITHM_VERSION
from ._stable.stable_v1 import assignment_from_output, group as _group


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BandGroupingResult:
    """Original input objects plus raw-candidate labels and ordinary geometry.

    ``band_ids[s][j]`` labels exactly input candidate ``(s, j)``.
    ``assignment[s, b]`` is its local input index, or -1 for a real absence.
    ``source_rows`` and height arrays have shape ``(bands, *grid_shape)``;
    missing positions are -1 and NaN, respectively. Triangle vertices index
    site-major, local-major original rows, never fitted or interpolated data.

    Arbitrary caller payload/descriptors remain on ``candidates`` unchanged
    and never select labels. Diagnostics are separate conditional evidence,
    not calibrated probabilities. No file I/O or renderer is used.
    """
    grid: ParameterGrid
    candidates: BandCandidates
    assignment: np.ndarray
    band_ids: tuple[np.ndarray, ...]
    source_rows: np.ndarray
    height_real: np.ndarray
    height_imag: np.ndarray
    triangles: np.ndarray
    triangle_band_ids: np.ndarray
    triangle_cells: np.ndarray
    diagnostics: Mapping[str, Any]

    @property
    def n_bands(self) -> int:
        return int(self.assignment.shape[1])

    @property
    def n_candidates(self) -> int:
        return self.candidates.n_candidates

    @property
    def algorithm_version(self) -> str:
        return ALGORITHM_VERSION


def _validate_contract(grid, candidates, config, initial_assignment):
    if not isinstance(grid, ParameterGrid) or not isinstance(candidates, BandCandidates):
        raise TypeError('use ParameterGrid and BandCandidates; adapt consumer objects explicitly')
    if not isinstance(config, GroupingConfig):
        raise TypeError('config must be a GroupingConfig')
    for name in ('max_model_roots', 'max_factors_per_block'):
        value = getattr(config, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + ' must be an integer, not a rounded resource limit')
    if grid.grid_shape is None or len(grid.grid_shape) not in (1, 2):
        raise ValueError('Stable v1 supports original one- or two-dimensional rectilinear grids')
    if len(grid.axis_values) != len(grid.grid_shape):
        raise ValueError('supply explicit axis_values (prefer ParameterGrid.regular_grid)')
    if any(not np.isfinite(a).all() or np.any(np.diff(a) <= 0) for a in grid.axis_values):
        raise ValueError('axes must be finite and strictly increasing')
    expected = ParameterGrid.regular_grid(*grid.axis_values)
    if (not np.array_equal(grid.coordinates, expected.coordinates)
            or grid.edges != expected.edges or grid.faces != expected.faces):
        raise ValueError('coordinates and topology must describe the full original rectilinear grid')
    if len(grid.grid_shape) == 1 and initial_assignment is None:
        raise ValueError('Stable v1 automatic scalar seeding is 2D; a 1D grid requires an explicit provisional initial_assignment')
    if candidates.n_sites != grid.n_sites:
        raise ValueError('candidate site count differs from the grid')


def group_bands(
    grid: ParameterGrid,
    candidates: BandCandidates,
    *,
    config: GroupingConfig | None = None,
    initial_assignment: np.ndarray | None = None,
    keep_models: bool = False,
) -> BandGroupingResult:
    """Group all original observations without adding, fitting or deleting any.

    Ordinary use supplies only a 2D grid and finite per-site complex arrays.
    Use an empty array for a genuinely empty site. Equal values are distinct
    observations and retain multiplicity. An optional integer site-by-band
    seed must partition every observation exactly once; it is not ground truth.

    Defaults are exactly Stable v1.0.0: epsilon_f=5e-5 in original frequency
    units, cubic local parameter models, at most 5 model roots, 1000 factors
    per block and 20 seconds per solver stage. These are method/resource
    boundaries, not physical uncertainty bounds. Unresolved local regions
    retain the provisional seed with explicit diagnostics; invalid inputs or
    execution errors raise instead of being relabeled as difficult physics.

    The internal modeling neighborhood is selected from the full-data seed;
    local holdout checks therefore remain conditional, not blind validation.
    Disconnected observable components may receive separate output IDs.
    """
    chosen = GroupingConfig() if config is None else config
    _validate_contract(grid, candidates, chosen, initial_assignment)
    raw, diagnostic = _group(grid, candidates, config=chosen,
        initial_assignment=initial_assignment, keep_models=keep_models)
    assignment = assignment_from_output(grid, raw)
    labels = [np.full(len(z), -1, dtype=int) for z in candidates.values]
    for row in raw.rows:
        labels[row['site']][row['local']] = row['band_id']
    if any(np.any(row < 0) for row in labels):
        raise RuntimeError('shared grouping lost an original observation')
    diagnostic = deepcopy(diagnostic)
    diagnostic['implementation'] = {'distribution': 'eigenmode-analysis',
        'version': __version__, 'algorithm_version': ALGORITHM_VERSION}
    diagnostic['input_contract'] = 'original rectilinear grid and unordered complex candidate arrays'
    # Only numerical fields and topology become output arrays. Consumer payload
    # is kept on candidates; no special dictionary keys are interpreted here.
    return BandGroupingResult(grid, candidates,
        _readonly(assignment, int), tuple(_readonly(row, int) for row in labels),
        raw.source_rows, raw.height_real, raw.height_imag,
        _readonly([t['vertices'] for t in raw.triangles], int).reshape(-1, 3),
        _readonly([t['band_id'] for t in raw.triangles], int),
        _readonly([t['base_face'] for t in raw.triangles], int),
        MappingProxyType(diagnostic))
