# Phase 5: Consumer Development Workflow

Status: completed on 2026-08-06

Date: 2026-08-06

## Scope

- Separate stable daily consumption from editable shared development.
- Report the exact shared version, source, install mode, and interpreter.
- Give humans and agents an explicit route from an A/B requirement to a shared
  primitive without moving consumer schema or workflow policy.
- Run one non-mutating compatibility matrix across shared, A, and B.

## Decisions

Daily A/B analysis environments install stable wheels. Editable installs are
limited to isolated development venvs. Production code does not inject sibling
paths. Workbench's COMSOL Python remains independent and never receives an
editable shared install.

Shared owns schema-neutral numerics and runtime/install tooling. A owns its
DataFrame, TSV, object-grid, Visualizer, and batch adapters. B owns manifest,
Spec, Registry, COMSOL, persistence, and run-provenance adapters. Mixed
features are split along that boundary.

`eigenmode_analysis.runtime` reports version, install mode, module location,
interpreter, prefix, and PEP 610 direct URL. Both consumers expose this record
when the shared tracker runs. `tools/manage_install.py` requires an explicit
target interpreter and rejects system-Python mutation by default.

`tools/verify_consumers.py` tests the current shared checkout in both consumer
analysis environments without changing either installation. Its temporary
source path and package-absence blocker are test-harness mechanisms, not
production import mechanisms.

## Release 0.2.1

This patch release adds operational provenance and installation tooling without
changing the Phase 1-4 numerical contracts. Consumer dependency ranges remain
minor-version ranges; adapters retain a fallback record for older compatible
releases that predate `runtime_info()`.

## Verification

- Shared tests: 24 passed in both the myPlots and Workbench analysis
  environments.
- myPlots adapter tests: 10 passed with 206 pre-existing NumPy
  scalar-conversion warnings.
- Workbench with shared source: 80 passed with 7 pre-existing singularity
  warnings.
- Workbench with the optional package blocked: 74 passed, 1 skipped, with the
  same 7 warnings.
- Dependency-boundary scan: no consumer, pandas, plotting, MPh, or JPype imports
  under `src/eigenmode_analysis/`.
- Built `eigenmode_analysis-0.2.1-py3-none-any.whl` and verified a disposable
  venv reported `wheel -> editable -> wheel`; the final import returned to
  `site-packages`.
- The installer refused an editable mutation of
  `D:\Program Files\Python311\python.exe` without `--allow-system`.
