# Phase 11: selective numerical extraction

Release 0.2.10 moved only three approved schema-neutral bodies from myPlots:

- Lorentzian/Fano fitting;
- field-regime skeleton extraction and classification;
- angular-spectrum propagation and XZ slicing.

The extraction preserves array-oriented signatures and imports only NumPy,
SciPy, and the standard library. Plotting, DataFrames, file formats, global
style, and figure lifecycle remain outside this package.

This phase explicitly rejected a blanket core migration. The experimental
polynomial tracker and wavelength-angle experimental transform remain in their
original myPlots modules. The package suite passes under the myPlots analysis
interpreter, and the consumer ownership suite verifies the old compatibility
paths.
