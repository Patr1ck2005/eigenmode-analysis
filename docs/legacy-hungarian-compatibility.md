# Legacy Hungarian compatibility

`tracking.legacy_hungarian` contains an explicit compatibility implementation
for established candidate-grid workflows. It preserves NaN handling, rank and
frequency-window constraints, auxiliary-data selection, and auto-split
diagnostics while consumers migrate to the typed tracking APIs.

New code should prefer `tracking.path` or `tracking.graph` when their explicit
contracts are sufficient. Compatibility code remains array-oriented and does
not read files or project schemas.
