# Stable v1 raw-observation band grouping

`eigenmode_analysis.band_grouping` is an opt-in numerical API introduced in
**eigenmode-analysis 0.3.1**. Its algorithm policy remains **stable-v1.0.0**,
ported from the frozen myPlots research release rather than redesigned here.
The existing `tracking` path/graph APIs and their defaults are unchanged.

## Example

```python
import numpy as np
from eigenmode_analysis.band_grouping import (
    ParameterGrid, BandCandidates, GroupingConfig, group_bands,
)

kx = np.linspace(-1.0, 1.0, 7)
ky = np.linspace(-0.7, 0.7, 7)
grid = ParameterGrid.regular_grid(kx, ky, axis_names=('kx', 'ky'))
# Site-major, x-first ordering; each site holds its unordered raw candidates.
values = tuple(np.array([1 + .25*(x+.13) + .03*y + .02j,
                         1 - .25*(x+.13) + .03*y + .02j])
               for x, y in grid.coordinates)
result = group_bands(grid, BandCandidates(values), config=GroupingConfig())

labels = result.band_ids       # labels[s][j] belongs to original values[s][j]
assignment = result.assignment # [site, band] -> original local index; -1 absent
real_surfaces = result.height_real  # [band, kx_index, ky_index]; NaN absent
warnings = result.diagnostics  # separate evidence, not part of the raw values
```

`BandCandidates` can also retain per-site `original_indices`, arbitrary opaque
`payload` and numerical `descriptors`. They never participate in classification
and remain available on `result.candidates`. The shared package does not know
consumer column names or special payload keys. Consumers own all file formats,
DataFrame mapping, pickle trust, rendering, serialization and output naming.

## Contracts

- Automatic seeding is supported on full **two-dimensional rectilinear grids**
  with finite, strictly increasing axes. One-dimensional grids require an
  explicitly supplied provisional `initial_assignment`; no automatic 1D
  capability is inferred from an outer legacy guard.
- Observations are finite complex arrays, one array per original parameter
  site. An empty array means a genuinely empty site. Equal frequencies remain
  distinct observations with independent `(site, local_index)` identities.
- No input value is fitted, replaced, dropped or duplicated in output.
  Internal model roots and diagnostic interpolation points are not observations.
- `assignment` has shape `(sites, bands)`. Each original local candidate occurs
  exactly once in its row, and each band has at most one candidate at a site.
- `band_ids` mirrors the variable-length input arrays; `source_rows`,
  `height_real` and `height_imag` have shape `(bands, *grid_shape)`. Missing
  entries are `-1` or `NaN`, not zero-frequency placeholder observations.
- `triangles` has shape `(n_triangles, 3)`, indexing flattened site-major raw
  rows. `triangle_band_ids` and `triangle_cells` align with it. A quadrilateral
  produces the fixed halves `(0,1,2)` and `(0,2,3)` only when all four original
  candidates of that band exist. No plotting backend is imported.
- Exported array views are read-only. Disconnected observed components may
  receive separate band IDs, preserving the frozen exporter convention.
- `initial_assignment` is optional, integer, site-by-band, with `-1` absences.
  It is an unverified numerical seed, not a place to supply human truth labels.

## Fixed policy and scientific limits

Defaults are `epsilon_f=5e-5`, at most five model roots, at most 1000 original
axis/diagonal factors per block and 20 seconds per integer-solver stage. The
frequency tolerance is in original units; it is not a measured noise bound.
Local parameter polynomials are cubic. Complete and partial spectra follow the
frozen policies; nearby singleton bands may supply internal modeling context
without forcing output groups to merge.

Every observation receives a default grouping. Unsupported model families,
resource limits, unresolved numerical continuation and inconsistent conditional
constraints retain the provisional seed where required and are explicitly
reported. These states are **not** proof of an incorrect physical identity, nor
proof that the data are intrinsically impossible. Invalid inputs and execution
failures raise rather than being classified as difficult physics.

Conditional support is not a calibrated probability or physical correctness
certificate. The internal membership and modeling window use a full-data seed,
so local leave-out errors are conditional, not blind end-to-end validation.
Finite samples may admit both a crossing and an unresolved narrow avoided
crossing. Some continuation-reference loops cannot be represented by a single
`band_id` at each site. The known EP2 residual reference disagreements are not
removed or relabeled as globally correct by this library migration.

## Reproducibility and installation

The shared runtime needs only NumPy, SciPy and the standard library. Its private
implementation has no myPlots, benchmark, file-I/O or renderer dependency.
`docs/stages/phase-12-source-map.json` records the pinned numerical extraction.
The original `stable-v1.0.0.zip` remains immutable in the consumer repository.

The package version and algorithm version are distinct. Use
`eigenmode_analysis.runtime.runtime_info()` alongside
`result.algorithm_version` to record installation provenance. The cross-consumer
harness is the only development path permitted to inject exact source paths;
normal use installs a wheel, with no sibling source or PYTHONPATH dependency.

0.3.1 is built from this repository's public source line and the new grouping
API. Some older local-only wheels have unpublished extra modules not present
in this checkout. Integration therefore does not automatically replace a daily
consumer environment; validate the consumer's required APIs first.
