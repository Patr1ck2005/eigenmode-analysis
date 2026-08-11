# Phase 08: Jones, Stokes, and Poincare conversions

Release: `0.2.4`

## Scope

The new `eigenmode_analysis.polarization` module owns the reusable numerical
conversions previously distributed across myPlots utilities and Janus_BICs
project scripts:

- ellipse orientation/axis ratio to normalized Stokes;
- Stokes to ellipse parameters;
- Jones to Stokes using the established myPlots S3 handedness convention;
- fully polarized Stokes to a Jones representative;
- Stokes normalization and Poincare point generation.

Zero-intensity Jones fields and zero Stokes vectors have an explicit NaN/zero
policy instead of producing uncontrolled divide-by-zero warnings. Rendering
remains outside this NumPy-only package.

## Verification

- Shared package: 33 passed.
- Round trips cover linear, circular, and diagonal polarization.
- Zero-intensity and invalid Poincare samples are characterized.

## Consumer rollout

myPlots routes `utils.functions.ellipse2stokes`, `stokes2ellipse`, and the
Janus_BICs Jones/Stokes helpers through this shared module. Workbench exposes
the same functions through its analysis adapter and records shared runtime
provenance.
