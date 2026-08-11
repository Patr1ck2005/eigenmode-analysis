# Phase 4: N-D Graph Tracking and myPlots 2D Compatibility

Status: completed on 2026-08-06

Date: 2026-08-06

## Scope

- Add an explicit directed-acyclic traversal contract for N-D candidate data.
- Preserve the Phase 1 path behavior while sharing cost and anchor primitives.
- Diagnose multi-parent assignment disagreement instead of hiding it behind
  input edge order.
- Map myPlots' historical 2D object grid to the shared graph API without
  changing the legacy default backend.
- Audit active `auto_split_streams` workflows before deciding whether to move
  that heuristic out of the legacy tracker.

## Shared graph contract

The caller supplies every node, a topological `node_order`, all directed edges,
and the anchor. `rectilinear_traversal` constructs the historical positive-axis
origin traversal for dense grids. Sparse domains use explicit empty candidate
sets and/or explicit edge omission.

At a merge node, incoming costs are summed commutatively. Every informative
incoming edge also runs an independent proposal. Proposal disagreement is
returned in `GraphConflictDiagnostic`; the aggregate Hungarian result remains
available, but the conflict is never silently erased.

Disconnected nodes have reason `unreachable`. Empty reachable nodes remain
empty and can carry constant-window memory to later nodes. A track without a
finite anchor candidate cannot appear later.

`anchor_slope` is intentionally unsupported on a graph. Shortest-path,
longest-path, and per-branch distance give different acceptance regions, so the
package will not choose one implicitly. Constant `anchor_window` requires a
candidate to satisfy every reachable incoming branch.

## myPlots mapping

`group_vectors_nd_hungarian_shared` maps each object-grid cell to a
`CandidateSet` whose candidate ID is the original cell index. The result uses
those IDs to recover `additional_data`; it never matches metadata by floating
point value.

`track_vectors_nd_hungarian_shared` exposes the raw `GraphTrackingResult` to
myPlots callers that need merge conflicts, transition reasons, costs, or
unreachable-node diagnostics. The workflow compatibility functions retain
their historical return shape.

The compatible historical subset is:

- dimension count of at least two;
- origin anchor and positive-index rectilinear traversal;
- arbitrary numeric feature count, including complex features;
- diagonal cross-axis derivative matrix;
- the historical value matrix, whose row sum becomes the edge value scale;
- optional rank penalty, NaN cost, fixed window, fixed track count, and metadata
  selector.

The adapter rejects non-diagonal derivative matrices. For 2D `freq_window`, the
legacy implementation keeps one global last-accepted value in lexicographic
traversal order, while the graph contract has branch-local memory and requires
acceptance against every reachable incoming branch. This difference is public,
and every opt-in dataset must be revalidated. The `4a_real_data` regression is
numerically identical under both policies.

## auto-split decision

`auto_split_streams` is a myPlots postprocessor, not part of Hungarian
assignment. It cuts neighbor edges above a median-plus-MAD threshold, extracts
connected components, and reallocates them into empty stream slots. It can
discard short segments and therefore changes coverage after tracking.

It remains legacy-only in Phase 4. The active U-BIC defaults were evaluated
read-only against current data:

| Workflow | Grid | Assigned before/after | Changed cells |
| --- | ---: | ---: | ---: |
| U-BIC 1D NanoP | 61 | 183 / 183 | 0 |
| U-BIC 2D, t=186.45 nm | 31 x 31 | 3896 / 3896 | 0 |
| U-BIC 2D, t=189.25 nm | 31 x 31 | 3735 / 3735 | 0 |
| U-BIC 2D, t=200.165 nm | 31 x 31 | 3979 / 3979 | 0 |

Two historical direct-call scripts also enable it. The flatband case uses a
non-diagonal derivative matrix, which is already outside the shared adapter's
defined subset. Neither historical script has a maintained auto-split
regression contract. Extracting the postprocessor now would add migration risk
without exercising a case where it has an effect.

## Numerical equivalence evidence

The shared adapter matched legacy outputs exactly on all five existing
synthetic 2D benchmark families: regular surfaces, crossing ridges,
near-degenerate sheets, a missing patch, and mixed spurious candidates.

The existing `4a_real_data` regression also matched exactly:

| Metric | Shared and legacy |
| --- | ---: |
| Coverage | 0.9319240374005459 |
| Assigned | 240702 / 258285 |
| Continuity p95 | 0.0004252360137044644 |
| Maximum frequency jump | 0.0005322366825288089 |
| Regression gate | passed |

On the current machine, the full `4a` assignment took about 28.1 seconds on
legacy and 38.4 seconds on the shared graph. Correctness and diagnostics are the
Phase 4 gates; graph performance work must be a separate measured change.

## Verification

- Shared package tests: 21 passed in both the myPlots and Workbench analysis
  environments.
- myPlots shared adapter tests: 9 passed.
- Existing myPlots warning baseline: 206 NumPy scalar-conversion warnings in
  the 2D DataFrame compatibility test.
- myPlots existing Hungarian smoke: 1 passed.
- Workbench with the shared source available: 79 passed with 7 existing
  singularity warnings.
- Workbench without the optional package: 74 passed, 1 skipped, with the same
  7 warnings; base imports remained available.
- Wheel: `eigenmode_analysis-0.2.0-py3-none-any.whl`, imported successfully in
  both analysis environments.
- Dependency-boundary scan found no consumer, pandas, plotting, MPh, or JPype
  imports under `src/eigenmode_analysis/`.
