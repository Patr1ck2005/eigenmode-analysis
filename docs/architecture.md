# Architecture contract

`eigenmode-analysis` is the numerical layer in a reusable analysis stack:

```text
consumer schema/IO -> adapter -> eigenmode-analysis -> arrays/models
                                      |
                                      +-> plot-foundation (separate layer)
```

The package owns assignment costs, traversal, anchors, candidate identity,
missing-mode diagnostics, symmetry transforms, polarization conventions,
topological quantities, propagation, fitting, and grid sampling.

Consumers own file formats, DataFrame or manifest schemas, physical filtering,
output paths, logging, and workflow policy. Plot rendering is deliberately
independent and must not be imported here.

## Invariants

- Candidate IDs are explicit and are never recovered from floating-point value
  comparisons after tracking.
- Traversal order, anchors, coordinate distances, and rejection policy are
  explicit inputs.
- Near-degenerate assignments use minimum-cost one-to-one matching.
- Missing and rejected candidates remain represented as missing in results.
- Numerical diagnostics are part of the result contract.
- Consumers record runtime provenance when a shared result is used.

## Compatibility

Legacy helpers may remain when their array contract is useful to more than one
consumer. Consumer-specific wrappers preserve old call paths; they do not
become dependencies of this package.
