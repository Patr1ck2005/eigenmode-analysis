# Phase 6: Shared Grid Postprocessing

Status: implementation complete on 2026-08-06

Release: `0.2.2`

`split_disconnected_grid_tracks()` promotes myPlots' `auto_split_streams`
behavior into the schema-neutral numerical package. It works on explicit grid
node IDs, tracked candidate IDs, and feature arrays; consumers remain
responsible for DataFrame or manifest mapping.

The postprocessor preserves continuous streams, separates jump-disconnected
components by direct grid connectivity, and moves later components only into
empty track slots. NaN cells remain missing and candidate IDs move with their
features so metadata adapters cannot silently desynchronize.
