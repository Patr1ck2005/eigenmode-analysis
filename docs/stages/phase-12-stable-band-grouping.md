# Phase 12: Stable v1 band grouping in the shared numerical library

## Completed integration

**Local integration and installed-wheel verification are complete.** The public entry is `eigenmode_analysis.band_grouping.group_bands`; the myPlots adapter is `myplots_app.experimental.band_tracking.shared_grouping.group`. Package 0.3.1 keeps the frozen stable-v1.0.0 algorithm policy. The final receipt is [RELEASE.json](../../dist/stable-v1-integration-0.3.1/RELEASE.json); API and examples are in [band-grouping.md](../band-grouping.md).

The 103160-byte wheel is `dist/eigenmode_analysis-0.3.1-py3-none-any.whl` (SHA256 `86f6321637caef4d903feaef8a6284332b1c7e8430fb4eb785f63137283a4982`). Installed-wheel tests: 86 passed. Fresh frozen-input comparisons: all 6 exact. Separate myPlots real-input adapter replays: both exact, including original fields. The intentionally absent-library Workbench arm remains a recorded collection failure; the whole compatibility matrix is **not** claimed all-green.

## Scope and origin

User authorized integrating the frozen myPlots `stable-v1.0.0` algorithm into this repository. Algorithm behavior, raw observations, multiplicity, missing values, and scientific limitations remain unchanged. Public-library integration is not full physical qualification and does not authorize a public upload, tag, push, or a daily-environment switch.

- Shared Session: `wc_sess_000866f5103a4a80b092707e684e150b`; myPlots Session: `wc_sess_eac73afc921c401d8e23c6a2db2958cc`.
- Shared starting HEAD: `e172495aa0c561cf0edf251ae9828dd5e54d2d64`, branch `feature/band-tracking-v2`. Existing AGENTS.md and PROJECT_STATUS.md changes are not this task's writes.
- myPlots starting HEAD: `678cfe9a8d3a57f0b3abc374b3f3b3a06bf6e71d`. Frozen release archive SHA256: `794bf73f4cca56184f95575a5d493ac56822611c207820a32f92faba9709d67c`. Never overwrite that release.
- Existing dist contains 0.2.11 and 0.3.0 historical wheels, but the checkout is the 0.2.10 public source baseline. Use the new, unoccupied version 0.3.1; retain current source's legacy APIs. Do not silently install over the daily 0.2.11 environment, whose historical wheel contains extra unpublished modules.

## Implementation and validation history

1. Lift only the reachable Stable v1 numerical functions into `eigenmode_analysis.band_grouping`, preserving operation order. Remove imports of consumer applications, benchmark generators, renderers, manifests, and file I/O. Supply a typed array API and explicit diagnostics.
2. Keep a source-hash/extraction map from the immutable release. Add public contract tests and port the relevant existing numerical tests; include missing/multiple candidates, source isolation, unsupported cases and guard failures.
3. Build and install a versioned wheel in this repository's new isolated `.venv`; leave consumer environments untouched. Check NumPy/SciPy-only runtime imports and existing public APIs.
4. Add an opt-in thin myPlots adapter that forwards numeric inputs to the shared package and routes original fields through its existing exporter. Run frozen daily6 equivalence plus consumer compatibility checks, no new physics claim or full 519-case promotion.
5. Update API, changelog and both repositories' current entry/checkpoint, preserving archived source and unrelated files.

Initial review identified the runtime-only dependency closure; historical module-level imports were trimmed rather than copying the research harness wholesale. The following entries retain the bounded development history, not active Job instructions.

Isolated development-environment preparation Job: `3f426b84-207b-42d7-9a38-a09547993eeb`. Uses this repository's new `.venv` only; never changes myPlots/workbench environments. The source port is a create-only, archive-SHA-guarded extraction through `tools/port_stable_band_grouping.py`, not an unreviewed wholesale copy of consumer files.

Source integration now contains 17 explicitly extracted numerical modules (2626 lines), a pure ordinary-grid geometry exporter, and the public array API. No runtime import of consumer or benchmark namespaces remains. The 19 existing spectral/partial-relation tests have been ported into the shared suite. Public plus ported tests: **33 passed** (Job `f4d44784-3957-41ef-9ef1-0a5bac87e9fe`). One initial test harness incorrectly blocked NumPy's standard-library pickle import; that harness was corrected while the static no-direct-I/O check remains. No classifier change was needed.

Editable install was verified only in the new shared `.venv` (Job `1bd0ed8e-b136-42bf-a612-61c491c90683`). myPlots now has an opt-in `shared_grouping` adapter and five characterization tests; it retains legacy field routing after the shared numeric call. Next: full compatibility matrix, clean wheel build, installed-wheel frozen daily6 verification. Frozen release and both daily environments remain untouched.

Completed Jobs: compatibility matrix `85ba9aef-fcdd-4794-925f-56cf1d9cb2d5` (log/report under `dist/stable-v1-integration-0.3.1/`); clean versioned wheel build `a0b72148-a7fc-4069-bd31-e62545ca0b91`. The wheel is built from a temporary exact source snapshot, so stale files in the pre-existing build directory cannot leak into the package. Before restarting anything, inspect these Job IDs and saved reports.

Build completed: `dist/eigenmode_analysis-0.3.1-py3-none-any.whl`, 103160 bytes, SHA256 `86f6321637caef4d903feaef8a6284332b1c7e8430fb4eb785f63137283a4982`; all 40 package Python files byte-equal to the clean source snapshot. Installed as a non-editable wheel in the shared `.venv` (Job `f3d5e7b8-8303-423b-9c38-4c3dd4b9b4af`). The installed-wheel suite now passes **86 tests**; the two old runtime tests were corrected to verify the actual source/wheel mode and reject its opposite, rather than always assuming source-tree mode. Runtime classifier sources and wheel did not change.

Compatibility results: shared suite 86 passed in each of the myPlots and Workbench environments; myPlots adapters 69 passed; Workbench with shared library 352 passed. The explicitly absent-package matrix arm failed collection because its existing shared visualizer imports plot_workflows.tracking, which directly requires the deliberately blocked eigenmode_analysis.tracking. No new band-grouping code is entered in that arm. The full matrix remains marked partial_failure in `compatibility-readback.json`; no consumer sources were modified or failure silently skipped. Dependency-boundary scan passed independently.

Installed-wheel daily6 fresh replay Job `06f97b80-17ab-4fae-bf9c-2bd665613a7b` completed successfully; output `dist/stable-v1-integration-0.3.1/wheel-daily6/`. It has no consumer or historical-result code on the import path, and opens expected labels only after each fresh output is saved.

The installed-wheel daily6 Job completed successfully: **all six fresh classifications have exactly the same original labels, height grids, missing positions and ordered triangles as the frozen release**, with every original field preserved. C6v: 1256 observations / 10 bands / 44 missing slots / 2072 triangles. Chiral: 2366 / 14 / 0 / 4032. Historical EP2 reference disagreements are unchanged, not reclassified as physically correct.

Additional end-to-end myPlots adapter verification Job `2974426a-c3cf-4791-a882-d3850c89ff26` completed successfully in the installed-wheel environment; both full `bands.json` objects and original fields match the frozen release. Its compatibility harness exposes only the consumer root, not sibling shared source. Output: myPlots `tests/bandtrack/results/method-reviews/20260921-shared-library/wheel-adapter-real2/`.

## Current recovery point

At the end of integration, all required compute Jobs had finished. The integration-time receipt remains `dist/stable-v1-integration-0.3.1/RELEASE.json`; it records the pre-commit HEAD and is not rewritten to pretend the earlier build occurred after a commit. The shared `.venv` uses the installed 0.3.1 wheel, not editable source. Both daily consumer environments and myPlots requirements remain unchanged. Integration itself performed no public package upload, commit, push, tag or service change.

The frozen v1 archive SHA is unchanged. All 17 extracted numerical-source hashes and all 40 packaged Python files were rechecked; input fields, multiplicity and missing support are preserved. Current source has a traceable opt-in public interface, but no new 519-case scientific qualification or global physical uniqueness claim.

### Authorized Git closeout

After the tunnel recovered, the user explicitly authorized scoped Git commits with `Origin: ai:gpt-6-astra-pro-web-chat-mcp`. Rechecked the installed wheel against all 40 current packaged Python sources, every referenced validation-receipt hash, the unchanged frozen v1 archive, and the empty index. The 36 integration files are isolated from the two unrelated existing changes (`AGENTS.md`, `PROJECT_STATUS.md`), whose original hashes are preserved in `dist/stable-v1-integration-0.3.1/git-closeout-before.json`.

The final installed-wheel test Job `ecc76122-17f6-421a-a98a-4de8b06ec197` completed with 86 passed, with XML at `git-closeout-tests.xml` in the same evidence directory. Core API, numerical implementation, package version and tests were committed as `5cd2f77bea0dcaca78fa0608270ef19475568963`, with the requested Origin trailer. This follow-up records the pinned extraction, reproducibility tools and final handoff. Before resuming a disconnected closeout, inspect the saved output and actual Git log instead of launching the test or committing again. Final commit IDs and post-commit preservation checks are recorded in `git-closeout.json`; no push, tag, public upload or consumer-environment switch is authorized.
