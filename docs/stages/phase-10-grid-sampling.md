# Phase 10: grid and path sampling

Release 0.2.9 moved the reusable grid/path sampling body from myPlots into
`eigenmode_analysis.grid_sampling`.

The shared contract includes wrapped angles, cell-edge lookup, bilinear scalar
and pi-periodic angle interpolation, uniform path resampling, and sampling a
mapping of x-first fields along a path. Inputs are arrays and mappings only;
DataFrame columns and eigensolution workflow remain in myPlots.

Verification completed with 50 package tests, the myPlots adapter suite, and
stable-wheel provenance in both consumers.

