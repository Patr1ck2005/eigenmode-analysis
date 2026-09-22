"""Stable research v1: fresh raw-data grouping, no case names or saved results.

This is a versioned LOCAL RESEARCH entry, not the production default. Reuses
G10/G17/G18 rather than introducing a new model family. Historical probes are
not invoked; truth, payload and provenance cannot choose the grouping.
"""
from dataclasses import asdict, dataclass
from collections import Counter
from types import SimpleNamespace
import numpy as np
from .model import CandidateFibers
from .classification import validate_assignment, mesh_from_assignment
from .grouped_output import build_grouped_output
from .local_pair_stitching import preferences
from .multiband_blocks import interaction_blocks
from .local_spectral_contact import diagnose
from .partial_observations import diagnose_partial, partial_factor
from .partial_observation_grouping import solve_partial, actual_pairs
from .projected_connections import evidence_edge

VERSION = 'stable-v1.0.0'


@dataclass(frozen=True)
class Config:
    epsilon_f: float = 5e-5
    max_model_roots: int = 5
    max_factors_per_block: int = 1000
    seconds_per_solver_stage: float = 20.0
    # Resource boundaries, not a statistical confidence level or a quality score.

    def __post_init__(self):
        if (not np.isfinite(self.epsilon_f) or self.epsilon_f <= 0 or
                not 2 <= self.max_model_roots <= 5 or self.max_factors_per_block < 1 or
                not np.isfinite(self.seconds_per_solver_stage) or self.seconds_per_solver_stage <= 0):
            raise ValueError('invalid stable research configuration')


def original_factors(c):
    rows = [{'kind': 'axis', 'id': i, 'sites': list(e)} for i, e in enumerate(c.edges)]
    known = {tuple(sorted(e)) for e in c.edges}
    for i, face in enumerate(c.faces):
        e = tuple(sorted((face[0], face[2])))
        if e not in known:
            rows.append({'kind': 'diagonal', 'id': i, 'sites': list(e)})
            known.add(e)
    return rows


def assignment_from_output(c, output):
    rows = output.rows if hasattr(output, 'rows') else output['rows']
    n = len(output.height_real) if hasattr(output, 'height_real') else len(output['height_real'])
    a = np.full((c.n_sites, n), -1, int)
    for row in rows:
        if a[row['site'], row['band_id']] >= 0:
            raise ValueError('repeated site/band in grouped output')
        a[row['site'], row['band_id']] = row['local']
    return a


def evidence_state(reason):
    """Lack of support is never silently converted to a physical error/pass."""
    if 'conflict' in reason or 'no_feasible' in reason or 'empty_' in reason:
        return 'conditional_constraints_inconsistent'
    if 'unresolved' in reason or 'collision' in reason or 'multiroot' in reason:
        return 'numerical_continuation_unresolved'
    if 'budget' in reason or 'limit' in reason or 'oversized' in reason:
        return 'resource_or_method_limit'
    return 'model_evidence_insufficient'


def _validate_input(c, f):
    if c.grid_shape is None or len(c.grid_shape) not in (1, 2):
        raise ValueError('stable v1 requires an original 1D/2D regular grid')
    if f.n_sites != c.n_sites or any(len(face) != 4 for face in c.faces):
        raise ValueError('inconsistent raw grid and observations')
    if any(np.any(np.diff(axis) <= 0) for axis in c.axis_values):
        raise ValueError('grid axes must be strictly increasing; reorder explicitly before grouping')
    if f.n_candidates == 0:
        raise ValueError('no observations to group')


def _expanded_blocks(a, groups, max_roots):
    active = [list(g) for g in groups if len(g) > 1]
    owner = {b: i for i, g in enumerate(active) for b in g}
    rows = []
    for i, g in enumerate(active):
        proposed = list(g)
        if np.any(a[:, g] < 0):
            # One fixed adjacent-neighbor rule, not error-driven per-case search.
            for b in (min(g)-1, max(g)+1):
                if 0 <= b < a.shape[1] and b not in owner and len(proposed) < max_roots:
                    proposed.append(b)
        rows.append(sorted(proposed))
    # Avoid mutually rewriting a shared neighboring band. Conflicting expansions
    # retain the original blocks rather than arbitrarily giving one precedence.
    conflicts = {i for i in range(len(rows)) if any(
        set(rows[i]) & set(rows[j]) for j in range(len(rows)) if i != j)}
    for i in conflicts:
        rows[i] = active[i]
    return active, rows


def group(c, f, *, config=None, initial_assignment=None, keep_models=False):
    """Return GroupedRawOutput and a separate honest diagnostic dictionary.

    No sample id, reference labels, cache path, or physical descriptors are
    arguments to the numerical policy. An explicit initial assignment is an
    optional unverified seed, never ground truth. Ordinary use omits it.
    """
    config = Config() if config is None else config
    _validate_input(c, f)
    numeric = CandidateFibers(tuple(np.array(z, copy=True) for z in f.values))
    if initial_assignment is None:
        from .model_guided_transport import reconstruct
        seed_mesh, seed_info = reconstruct(c, numeric, channel='real')
        seed = assignment_from_output(c, build_grouped_output(seed_mesh))
        seed_source = 'fresh_G10_Re'
    else:
        seed = validate_assignment(c, numeric.values, initial_assignment).copy()
        seed_info = {'status': 'caller_supplied_unverified_seed'}
        seed_source = 'caller_assignment'
    seed = validate_assignment(c, numeric.values, seed)
    answer = seed.copy()
    pref = preferences(c, numeric.values)
    groups = interaction_blocks(seed, [], pref)
    original, blocks = _expanded_blocks(seed, groups, config.max_model_roots)
    factors = original_factors(c)
    details = []
    for block_index, (original_bands, bands) in enumerate(zip(original, blocks)):
        row = {'initial_interaction_bands': original_bands, 'model_bands': bands,
               'output_band_count_not_enlarged': True, 'status': 'retained_seed',
               'fallback': True, 'evidence_status': 'model_evidence_insufficient'}
        if len(bands) > config.max_model_roots or len(c.grid_shape) != 2 or min(c.grid_shape) < 5:
            row.update(reason='oversized_block_or_unsupported_grid', evidence_status='resource_or_method_limit')
            details.append(row)
            continue
        if len(factors) > config.max_factors_per_block:
            row.update(reason='factor_budget_exceeded', evidence_status='resource_or_method_limit')
            details.append(row)
            continue
        complete = bool(np.all(seed[:, bands] >= 0))
        members = [sorted(int(j) for j in v if j >= 0) for v in seed[:, bands]]
        labels = [[int(np.flatnonzero(seed[s, bands] == j)[0]) for j in ids] for s, ids in enumerate(members)]
        edge_records, models = [], []
        for factor in factors:
            sites = factor['sites']
            if complete:
                proxy = SimpleNamespace(edges=[sites], grid_index=c.grid_index,
                    site_index=c.site_index, grid_shape=c.grid_shape, coordinates=c.coordinates)
                model = diagnose(proxy, numeric.values, seed, bands, 0,
                    epsilon=config.epsilon_f, boundary='shifted')
                full = evidence_edge(c, numeric.values, seed, bands, sites, model,
                    epsilon=config.epsilon_f, protect_baseline=False)
                s, t = sites
                left = {int(j): i for i, j in enumerate(members[s])}
                right = {int(j): i for i, j in enumerate(members[t])}
                options = [{'pairs': sorted([[left[int(seed[s, bands[i]])], right[int(seed[t, bands[j]])]]
                    for i, j in enumerate(o['mapping'])]), 'cost': o['cost']} for o in full['options']]
                edge = {k: v for k, v in full.items() if k != 'options'}
                edge['options'] = options
            else:
                model = diagnose_partial(c, numeric.values, members, sites, len(bands),
                    epsilon=config.epsilon_f)
                edge = partial_factor(c, numeric.values, members, seed, bands, sites, model,
                    epsilon=config.epsilon_f)
                if edge.get('supported') and len(edge.get('admitted_match_counts', [])) > 1:
                    # Avoid exploiting fewer observed paths to reduce drawing cost.
                    edge.update(supported=False, frozen_fallback=True,
                        reason='variable_observed_coverage_not_comparable',
                        options=[{'pairs': edge['baseline_pairs'], 'cost': 0.0}])
            edge.update(factor)
            edge['evidence_status'] = ('conditional_model_supported' if edge['supported'] else
                evidence_state(edge.get('reason', model['status'])))
            edge_records.append(edge)
            if keep_models:
                models.append(model)
        solved, solver = solve_partial(list(map(len, members)), len(bands), edge_records,
            labels, epsilon=config.epsilon_f, seconds=config.seconds_per_solver_stage)
        optimal = solver.get('primary_optimal', False) and solver.get('secondary_optimal', False)
        # This release favors a repeatable seed over a timing-dependent incumbent.
        accepted = solved is not None and optimal
        if accepted:
            for s, positions in enumerate(solved):
                answer[s, bands] = [members[s][j] if j >= 0 else -1 for j in positions]
        supported_count = sum(e['supported'] for e in edge_records)
        accepted_state = ('conditional_model_supported' if supported_count == len(edge_records) else
            'mixed_model_support' if supported_count else 'provisional_seed_only')
        row.update(status='conditional_grouping' if accepted else 'retained_seed',
            reason=None if accepted else solver['status'], fallback=not accepted,
            evidence_status=accepted_state if accepted else evidence_state(solver['status']),
            model_family='centered_cubic_complete' if complete else 'monic_observed_roots_cubic',
            solver=solver, factors=edge_records,
            factor_status_counts=dict(Counter(e['evidence_status'] for e in edge_records)),
            initial_assignment_role='conditional membership, not independent end-to-end holdout')
        if keep_models:
            row['models'] = models
        details.append(row)
    validate_assignment(c, f.values, answer)
    # Every original field is routed from f, not from the stripped numerical copy.
    output = build_grouped_output(mesh_from_assignment(c, f, answer, {'version': VERSION}))
    diagnostic = {'version': VERSION, 'config': asdict(config), 'seed_source': seed_source,
        'seed_status': seed_info.get('status'), 'blocks': details,
        'unmodeled_singleton_bands': [g[0] for g in groups if len(g) == 1 and not any(g[0] in b for b in blocks)],
        'raw_contract': {'observations': len(output.rows), 'groups': len(output.height_real),
            'triangles': len(output.triangles), 'missing_slots': int(np.count_nonzero(output.source_rows < 0))},
        'software_status': 'completed', 'scientific_status': 'conditional_or_provisional',
        'physical_correctness_proven': False, 'probability': None,
        'limits': ['Finite samples may admit crossing and narrow avoided-crossing extensions.',
                   'Unsupported numerical/model regions retain a documented provisional seed.',
                   'Solver/model support is not physical truth; no calibrated probability is supplied.',
                   'Disconnected observable parts may receive distinct export IDs under the existing exporter.'],
        'generated_observations': 0, 'deleted_parameter_factors': 0,
        'production_default_changed': False}
    return output, diagnostic
