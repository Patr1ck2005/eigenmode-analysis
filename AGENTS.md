# Agent instructions

This package provides reusable, domain-neutral eigenmode numerics.

## Boundary

- Runtime dependencies are NumPy, SciPy, and the Python standard library.
- Do not import plotting backends, DataFrames, simulator APIs, manifests, or
  consumer applications from `src/eigenmode_analysis/`.
- Public inputs and outputs are arrays and typed models with explicit identity,
  coordinates, traversal, and diagnostics.
- Missing or rejected modes remain missing; do not silently substitute a mode.

## Development

1. Characterize a numerical requirement with synthetic and consumer-side tests.
2. Add the schema-neutral implementation and package tests here.
3. Update consumer adapters and provenance after the shared contract is tested.
4. Use stable wheels in consumer environments; editable installs are limited to
   an isolated shared-development environment.
5. Update the public API or contract document for every release.
6. Run the package suite and dependency-boundary checks before tagging.
