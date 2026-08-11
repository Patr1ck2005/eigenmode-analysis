# Phase 2: myPlots 1D Compatibility

Status: completed on 2026-08-05

## Scope

- Map the historical myPlots 1D object-grid contract to schema-neutral
  `CandidateSet` inputs.
- Preserve candidate metadata through explicit candidate IDs.
- Add an opt-in `method="shared_hungarian"` backend to `run_pipeline_1d` and
  `run_pipeline_1d_from_grid` without changing the legacy default.

## Equivalence coverage

The compatibility suite compares complete numerical arrays with `equal_nan`
and compares mapped metadata candidate by candidate. Covered cases:

- smooth complex bands with shuffled solver ordering;
- missing mode segments and a non-finite candidate;
- anchor-window and rank-penalty policies;
- an end-to-end pandas pipeline using English `f_thz`, `Q`, and `band_idx`
  columns.

All four myPlots compatibility tests passed. The shared package's 12 tests
also remained green in the myPlots environment.

## Known baseline issue

The end-to-end pipeline emitted 126 existing NumPy deprecation warnings from
`core.process_multi_dim_params_space.group_solution`. Warning cleanup is a
separate maintenance task because changing scalar conversion is not required
for tracking equivalence.

## Deferred

- Changing the myPlots default backend.
- N-dimensional object-grid mapping.
- `auto_split_streams=True` compatibility.
- Packaging myPlots and declaring the released shared dependency.
