# Delegation Roadmap

Phases 1-3 established the package boundary, the difficult 1D numerical core,
myPlots compatibility, and the ResearchAgentWorkbench scalar adapter. The work
below is intentionally split into independent, reviewable batches. Each batch
must update its listed document before it is considered complete.

Phase 5 established how those consumers continue to drive shared development:
stable wheels for daily work, isolated editable environments for shared work,
runtime source/version provenance, and one compatibility matrix. See
`docs/consumer-development-workflow.md`.

## WP1: Shared N-D traversal contract

Status: completed in Phase 4 on 2026-08-06.

Owner profile: numerical/API agent.

Scope: design explicit traversal edges and anchors for rectangular and sparse
N-D candidate grids. Keep assignment local to edges and make merge conflicts
diagnostic rather than silently choosing traversal order. Do not import either
consumer.

Acceptance: synthetic 2D crossings, holes, NaNs, disconnected regions, and
order-permutation tests; documented conflict policy; no regression in 1D.

Documentation: add a new shared-package phase document and update
`docs/architecture.md`.

## WP2: myPlots 2D adapter and auto-split decision

Status: completed in Phase 4 on 2026-08-06. The 2D adapter is opt-in. Active
auto-split workflows remain legacy-only pending a characterization case where
the postprocessor materially changes output.

Owner profile: myPlots compatibility agent. Depends on WP1.

Scope: map the historical 2D object grid to the shared traversal contract and
compare full numeric/metadata outputs. Characterize actual usage of
`auto_split_streams`; either implement it as an explicit postprocessor or
deprecate it with evidence. Do not change the default backend in this batch.

Acceptance: existing 2D Hungarian cases and at least one real-data regression;
no output-shape or metadata-selector changes.

Documentation: update `docs/core-systems/eigenmode-analysis-migration.md` in
myPlots.

## WP3: myPlots plotting boundary

Status: moved to the independent `plot-foundation` repository on 2026-08-06.
The earlier same-distribution implementation was withdrawn before release.

Owner profile: architecture agent. Independent of WP1.

Scope: separate plotting internally into data payloads, rendering primitives,
Visualizer orchestration, and project wrappers. Eigenmode calculations may
produce payloads but must not import matplotlib or Visualizers. Preserve the
current Chinese DataFrame schema at the adapter boundary.

Start as an internal myPlots boundary. Extract a separate distribution only
after B or another project uses the same rendering API in real code; otherwise
the extra package adds release overhead without proven reuse.

Acceptance: import graph recorded before and after; representative plots are
pixel/data equivalent; existing project entry scripts remain thin wrappers.

Documentation: create `docs/core-systems/plotting-architecture.md` in myPlots.

## WP4: Packaging and bulk import migration

Owner profile: packaging agent. Depends on stable WP2/WP3 boundaries.

Scope: give myPlots an installable package boundary, declare
`eigenmode-analysis>=0.2,<0.3`, replace project-level path manipulation with
package imports, and provide a mechanical import mapping table. Migrate in
small batches without renaming Chinese columns.

Acceptance: clean-environment install, import smoke tests, rollbackable batches,
and unchanged plotting outputs.

Documentation: installation and migration guide in myPlots.

Phase 5 does not make myPlots itself installable and therefore does not mark
this work package complete. It provides only the shared-package wheel workflow
and consumer compatibility gate that WP4 will rely on.

## WP5: Symmetry extraction

Owner profile: numerical geometry agent. Depends only on Phase 1 contracts.

Scope: extract C2/C4/C6 coordinate and field transformation primitives into
the shared package. Keep DataFrame/grid reconstruction in consumer adapters.

Acceptance: group closure, inverse, coordinate tolerance, duplicate-point, and
consumer equivalence tests.

Documentation: shared symmetry contract plus A adapter notes.

## WP6: Polarization and Poincare split

Owner profile: physics/numerical agent.

Scope: place Jones/Stokes/Poincare calculations and conventions in the shared
package; keep colormaps, sphere rendering, labels, and figure composition in
the plotting layer. State normalization, handedness, and invalid-intensity
behavior explicitly.

Acceptance: analytic polarization states, convention tests, NaN/zero-intensity
tests, and A output equivalence.

Documentation: shared convention document and myPlots rendering adapter notes.

## WP7: Optional Workbench rendering backend

Status: basic line overview is being migrated to the independent
`plot-foundation` distribution. Surface, field, polarization, and Output Spec
expansion remain future work packages.

Owner profile: Workbench integration agent. Depends on WP3 and a demonstrated
second consumer.

Scope: adapt DatasetRegistry/Spec payloads to the reusable plotting API. Keep
Output Spec, output paths, and provenance in Workbench.

Acceptance: raw/tracked band figures for line datasets, no hardcoded field
variables, and Output Spec regression tests.

Documentation: Workbench output-backend integration guide.

## WP8: Maintenance cleanup

Owner profile: maintenance agent. Independent.

Scope: remove the 126 NumPy scalar-conversion warnings in
`group_solution` with characterization tests. Do not combine this with tracking
or plotting migrations.

Acceptance: zero warning for the English-schema compatibility test and no
numeric/shape change.

Documentation: append the verification result to the myPlots migration note.
