# Phase 3: ResearchAgentWorkbench Scalar Adapter

Status: completed on 2026-08-05

## Scope

- Map English manifest records to schema-neutral candidates.
- Select the path point nearest Gamma as an explicit bidirectional anchor.
- Preserve raw `band_idx` as candidate identity.
- Persist both the tracking map and tracked manifest atomically.
- Keep the historical two-argument custom-tracker hook compatible while
  allowing project defaults and Analysis Spec overrides.
- Add a path-distance-scaled anchor guard that is independent of scan
  resolution.
- Keep tracks absent at Gamma inactive instead of inventing a later mode birth.

## Data-based decision

The current `TEST-1X50` manifest contains 40 populated k-points, 2-4 physical
candidates per populated point, a four-candidate Gamma anchor at `k_idx=25`,
and a 267-428 THz frequency range. Therefore:

- the analytical-membrane trial tracks four Gamma modes, matching its theory
  contract;
- the adapter carries empty/missing candidate points honestly;
- the guard is a slope in THz per cumulative `(m1, m2)` path-coordinate unit;
- normalized-frequency defaults from myPlots are not reused.

## Calibration evidence

The read-only calibration command is:

```powershell
python projects/analytical-membrane/scripts/evaluate_scalar_tracking.py
```

The 13 frozen 201-point datasets determine continuity behavior. `TEST-1X50`
checks the same policy on a coarser 51-point scan. Field-overlap agreement is a
secondary reference normalized by the Gamma labels; it is not ground truth at
exact or near degeneracy.

| Slope | Mean coverage | Worst adjacent jump (THz) | Mean field agreement | TEST coverage | TEST jump (THz) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 0.670 | 1.765 | 0.658 | 0.627 | 5.336 |
| 300 | 0.695 | 1.990 | 0.657 | 0.652 | 5.934 |
| 325 | 0.701 | 2.175 | 0.663 | 0.667 | 6.494 |
| **350** | **0.704** | **2.402** | **0.664** | **0.672** | **6.563** |
| 375 | 0.708 | 2.650 | 0.657 | 0.672 | 6.563 |
| 400 | 0.711 | 2.822 | 0.658 | 0.672 | 6.563 |
| 500 | 0.716 | 3.435 | 0.658 | 0.672 | 6.563 |

At slope 350 the coarse-grid coverage has reached its plateau. Larger values
increase the permitted fine-grid jump without improving that validation
coverage. With no guard, the frozen-dataset mean coverage rises to 0.771 but
the worst adjacent jump reaches 114.000 THz, so unguarded scalar tracking is
not an acceptable production default.

## Configuration decision

The analytical-membrane project default is `n_tracks: 4` and
`anchor_slope: 350.0`. Formal runs can freeze or override the resolved policy
under `mode_tracking.options` in an Analysis Spec. The current trial spec is
`data/specs/analysis/scalar_tracking_trial.yaml`.

## Verification

- Shared package: 12 passed in both analysis environments.
- myPlots shared compatibility: 4 passed; 126 existing NumPy warnings.
- Workbench with the optional package: 79 passed, 7 existing warnings.
- Workbench without the optional package: 74 passed, 1 skipped, 7 existing
  warnings; base imports remained available.
- Wheel rebuilt as `eigenmode_analysis-0.1.0-py3-none-any.whl`.
