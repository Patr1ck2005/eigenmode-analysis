# eigenmode-analysis

`eigenmode-analysis` provides schema-neutral numerical primitives for
eigenvalue and mode analysis. It accepts arrays, coordinates, and typed models;
it does not read simulator files, DataFrames, project manifests, or plotting
configuration.

## Scope

- path and directed-graph mode tracking with explicit candidate identity;
- Hungarian assignment, missing-mode and NaN handling, and diagnostics;
- symmetry, polarization, topology, propagation, fitting, and grid sampling
  primitives;
- compatibility helpers for established candidate-grid workflows.

Plotting belongs to `plot-foundation`; high-level composition belongs to
`plot-workflows`; campaign file management belongs to `campaign-data`.

## Quick start

```python
from eigenmode_analysis.tracking import TrackingConfig, track_path

result = track_path(
    points,
    coordinates=k_values,
    anchor_index=anchor_index,
    config=TrackingConfig(),
)
```

For a rectilinear candidate grid:

```python
from eigenmode_analysis.tracking import (
    TrackingConfig, rectilinear_traversal, track_graph,
)

order, edges = rectilinear_traversal(shape=(31, 31), deltas=(0.001, 0.001))
result = track_graph(
    points_by_index,
    node_order=order,
    edges=edges,
    anchor_id=(0, 0),
    config=TrackingConfig(),
)
```

Fields use explicit x-first shapes where documented; no implicit spacing or
column-name convention is assumed.

## Installation and development

```powershell
python -m pip install eigenmode-analysis
python -m pip install "eigenmode-analysis[dev]"
python -m pytest -q
```

See [docs/public-api.md](docs/public-api.md) for the public boundary and
[docs/architecture.md](docs/architecture.md) for dependency rules.

## Versioning

Numerical contract changes require characterization tests and a changelog
entry. Stable consumer environments should install a tagged wheel.
