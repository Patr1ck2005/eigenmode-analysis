# Phase 9: Topological Textures

## Scope

Release 0.2.5 moves the reusable Skyrmion density and integration calculation
out of myPlots. The shared contract is array-only and uses the established
x-first orientation:

```text
field.shape == (len(x), len(y))
axis 0 == x
axis 1 == y
```

`skyrmion_density` evaluates `S dot (dS/dx cross dS/dy)` with explicit physical
coordinates. Optional Stokes normalization uses the package's existing NaN
policy for zero vectors. `integrate_skyrmion_density` uses the supplied
coordinates and rejects non-finite values inside the integration mask.

## Consumer boundary

The shared functions do not print, plot, select a region, infer coordinates,
or know DataFrame fields. myPlots keeps its historical function names as thin
adapters and owns any diagnostic figure. Workbench may map QuantityData arrays
to the same numerical contract without importing myPlots.

## Verification

- Non-square, non-uniform x/y grid orientation and sign.
- Explicit normalized versus raw Stokes behavior.
- Zero-vector NaN behavior.
- Coordinate-aware integration, masks, and non-finite rejection.
- Transposed-field and non-monotonic-axis rejection.
- Complete package suite: 37 passed under the myPlots analysis interpreter.
- Integration uses `scipy.integrate.trapezoid` so the same wheel supports A's
  NumPy 1.x and B's NumPy 2.x environments.
