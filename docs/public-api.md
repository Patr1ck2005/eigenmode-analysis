# Public API guide

The supported public surface is exported from `eigenmode_analysis` and its
documented submodules:

- `tracking`: candidate models, path/graph assignment, traversal, and
  compatibility tracking;
- `symmetry`: coordinate and field transforms;
- `polarization`: Jones/Stokes and related array conversions;
- `topology`: winding and texture quantities;
- `propagation`: angular-spectrum and field propagation helpers;
- `grid_sampling`, `lineshapes`, `peak_tracking`, and `field_regimes`:
  schema-neutral analysis primitives.

- `band_grouping` (0.3.1): `ParameterGrid`, `BandCandidates`, `GroupingConfig`,
  `BandGroupingResult`, `ALGORITHM_VERSION`, and `group_bands`; raw-observation
  Re-surface grouping under the frozen Stable v1 policy. See
  [band-grouping.md](band-grouping.md) for array shapes, absence/multiplicity,
  ordinary geometry and conditional evidence. Internal `_stable` modules
  are not a supported public API.

The API accepts arrays, coordinates, and typed configuration. Consumers are
responsible for reading files, mapping names, selecting physical modes, and
storing provenance.
