# Phase 6: Independent Plotting Split

Status: completed on 2026-08-06

## Correction

The initial Phase 6 implementation placed schema-neutral plotting under
`eigenmode_analysis.plotting`. That commit was withdrawn before release because
import isolation alone did not provide independent ownership, versioning, or
release cadence.

The plotting implementation and its tests moved to the independent
`plot-foundation` distribution. `eigenmode-analysis` returned to version 0.2.1
and its strict NumPy/SciPy-only dependency boundary. The two packages do not
depend on one another; myPlots and ResearchAgentWorkbench consume them directly.

The withdrawn state remains recoverable at `archive/phase6-single-package` for
audit purposes, but it is not part of the active feature branch.

## Verification gate

- No plotting package exists under `src/eigenmode_analysis`.
- Matplotlib remains a forbidden import in the dependency scan.
- The numerical suite passes in both consumer analysis environments.
- Consumer plotting acceptance is owned by `plot-foundation` and consumer phase
  documents, not by this numerical distribution.

