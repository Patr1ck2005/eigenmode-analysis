# eigenmode-analysis

<p align="center"><img src="docs/assets/portfolio/scientific-tools-v1.png" width="760" alt="Four glass modules linked by light: the shared scientific-tooling collection" /></p>

*Concept illustration for the shared scientific-tooling collection; not a computed result. [Artwork provenance](docs/assets/portfolio/manifest.json).*

For tasks in the personal research system, read the [canonical MyPhysics entry](D:/Obsidian/MyPhysics/System/README.md) and follow its pointers to the owning scientific project. This does not change the domain-neutral package contract. [Local Agent rules](AGENTS.md); [shared-library ownership](D:/Dev/Projects/Work/research-agent-workbench/docs/shared-libraries/shared-library-catalog.md).

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

## Real-frequency band-surface grouping

<p align="center"><img src="docs/assets/portfolio/band-grouping-v1.png" width="760" alt="Band grouping concept: discrete samples arranged into three coloured surfaces" /></p>

*Concept illustration, not computed data or a claim of unique physical mode identity. [Artwork provenance](docs/assets/portfolio/manifest.json).*

The opt-in `band_grouping` API preserves all unordered raw complex-frequency
observations and adds band IDs for ordinary Re surfaces:

```python
from eigenmode_analysis.band_grouping import (
    ParameterGrid, BandCandidates, group_bands,
)

result = group_bands(
    ParameterGrid.regular_grid(kx, ky),
    BandCandidates(values_by_site),
)
labels = result.band_ids
surfaces = result.height_real
```

The algorithm policy is the frozen `stable-v1.0.0`; the package release is
0.3.1. Genuine missing samples and equal-frequency multiplicity survive.
Model/solver limitations are reported separately from raw data, and no
probability or complete physical correctness claim is attached. Existing
path/graph tracking defaults do not change. See [the full array contract and
limits](docs/band-grouping.md), including the 1D explicit-seed restriction.

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
