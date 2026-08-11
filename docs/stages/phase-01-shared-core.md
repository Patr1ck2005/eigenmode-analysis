# Phase 1: Shared Tracking Core

Status: completed on 2026-08-05

## Scope

- Establish an independent, minimal-dependency Python package.
- Define schema-neutral candidate, configuration, result, and diagnostic models.
- Implement explicit-anchor bidirectional tracking for one-dimensional paths.
- Characterize the pre-extraction behavior of both consuming repositories.

## Baselines

Recorded on 2026-08-05 before shared-core integration:

- ResearchAgentWorkbench: 72 tests passed, with 7 existing singularity warnings.
- myPlots Hungarian pytest smoke test: 1 passed.
- myPlots `default-only` synthetic suite: 8 passed and 2 existing failures.
  The existing failures are `1d_true_crossing` and `1d_near_degenerate`.

The two myPlots failures are baseline limitations, not extraction regressions.
Phase 2 must preserve per-case metrics before a separate improvement phase
changes numerical behavior.

## Deferred

- N-dimensional legacy-grid compatibility.
- myPlots DataFrame and object-grid adapters.
- ResearchAgentWorkbench manifest and Gamma-path adapters.
- Mode birth that creates new tracks after the anchor.
- Symmetry, polarization, and plotting integration.

## Verification

- Package tests under the myPlots analysis environment: 12 passed.
- Package tests under the ResearchAgentWorkbench analysis environment: 12 passed.
- Wheel build: `eigenmode_analysis-0.1.0-py3-none-any.whl` created successfully.
- Dependency-boundary scan found no consumer, plotting, dataframe, or COMSOL
  imports under `src/eigenmode_analysis/`.

The completed core now includes a resolution-invariant `anchor_slope` guard,
last-accepted-coordinate recovery across missing segments, and an explicit
prohibition on tracks appearing without a finite anchor candidate.
