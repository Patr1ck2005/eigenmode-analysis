# Phase 07: Symmetry completion primitives

Release: `0.2.3`

## Scope

This phase promotes the mature myPlots momentum-space numerical helpers into
the schema-neutral `eigenmode_analysis.symmetry` module:

- scalar/complex geometric completion (`geom_complete`);
- C2, C4, and C6 polarization completion with the established `phi`/`chi`
  mirror and handedness sign conventions;
- C4 complex-spectrum completion.

The shared functions preserve A's `(len(kx), len(ky))` array orientation and
return one-dimensional coordinate axes. They do not import pandas, Matplotlib,
or consumer-specific schemas.

## Consumer migration

myPlots' `core.data_postprocess.momentum_space_toolkits` keeps the historical
function names but resolves them to this shared implementation when
`eigenmode-analysis>=0.2.3` is installed. The old definitions remain only as
a temporary compatibility fallback for an older wheel during rollout.

ResearchAgentWorkbench can import the same primitives directly from the shared
wheel; manifest and English-schema adaptation remains a Workbench concern.

## Verification

- Shared package tests: 29 passed with the symmetry characterization cases.
- myPlots shared tracking/plotting regression: 18 passed with the shared source
  checkout selected explicitly.
- No plotting backend is imported by the shared symmetry module.

## Follow-up

Next phases will migrate Jones/Stokes/Poincare numerical conversion and then
the corresponding polarization ellipse/Poincare rendering primitives into
Plot Foundation. The temporary A fallback can be removed only after the
stable-wheel consumer matrix covers all affected project scripts.
