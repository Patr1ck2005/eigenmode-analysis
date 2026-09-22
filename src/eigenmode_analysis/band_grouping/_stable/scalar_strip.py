"""Scalar-only strip reconstruction with delayed matching decisions.

The objective charges slope changes over complete sampled lines, not the fit
of one point-cloud chart. A beam retains competing partial matchings; both
traversal directions are evaluated with the same reversible objective.
Original scalar samples are the only numerical input to this module's solver.
Physical payloads and the unused frequency projection never enter it.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from .model import AssociationEdge, BandCoverMesh, CandidateRef, CoverFace
from .tracking import _assemble_components


@dataclass(frozen=True)
class StripConfig:
    beam: int = 16
    alternatives: int = 4
    speed_weight: float = .025
    unmatched_cost: float = 2.
    objective: str = 'bend'


def scalar_scale(values):
    gaps = np.concatenate([np.diff(np.sort(z)) for z in values if len(z) > 1]) \
        if any(len(z) > 1 for z in values) else np.array([])
    gaps = gaps[gaps > 1e-12]
    if len(gaps):
        return max(float(np.median(gaps)), 1e-10)
    steps = [abs(a[0]-b[0]) for a, b in zip(values, values[1:]) if len(a) == len(b) == 1]
    return max(float(np.median(steps)) if steps else .01, 1e-10)


def partial_match(cost, unmatched_cost, forbidden=()):
    n, m = cost.shape
    if not n or not m:
        return (), unmatched_cost*(n+m)
    aug = np.full((n+m, n+m), 1e30)
    aug[:n, :m] = cost
    for i, j in forbidden:
        aug[i, j] = 1e30
    aug[np.arange(n), m+np.arange(n)] = unmatched_cost
    aug[n+np.arange(m), np.arange(m)] = unmatched_cost
    aug[n:, m:] = 0.
    a, b = linear_sum_assignment(aug)
    pairs = tuple((int(i), int(j)) for i, j in zip(a, b) if i < n and j < m)
    value = sum(float(cost[i, j]) for i, j in pairs)+unmatched_cost*(n+m-2*len(pairs))
    return pairs, value


def matching_proposals(cost, config):
    best, score = partial_match(cost, config.unmatched_cost)
    rows = {best: score}
    # Forbidding a selected pair can reveal a joint reassignment of many rows;
    # it is not restricted to swapping the two closest branches.
    for pair in best:
        proposal, value = partial_match(cost, config.unmatched_cost, (pair,))
        rows[proposal] = value
    return sorted(rows.items(), key=lambda row: (row[1], row[0]))[:config.alternatives]


def transition_cost(x, values, step, previous, scale, config):
    a, b = values[step-1], values[step]
    delta = b[None, :]-a[:, None]
    cost = config.speed_weight*(delta/scale)**2
    if step > 1:
        h0, h1 = x[step-1]-x[step-2], x[step]-x[step-1]
        for i, j in previous:
            slope = (a[j]-values[step-2][i])/h0
            if config.objective == 'angle':
                step_scale = float(np.median(abs(np.diff(x))))
                first = np.array([h0/step_scale, (a[j]-values[step-2][i])/scale])
                second = np.column_stack([np.full(len(b), h1/step_scale), delta[j]/scale])
                cosine = second@first/(np.linalg.norm(second, axis=1)*np.linalg.norm(first))
                cost[j] += 2*(1-np.clip(cosine, -1., 1.))
            elif config.objective == 'bend':
                bend = (delta[j]/h1-slope)*(.5*(abs(h0)+abs(h1)))/scale
                cost[j] += bend*bend
            else:
                raise ValueError('unknown strip objective')
    return cost


def line_objective(x, values, path, scale, config):
    total = 0.
    for step, pairs in enumerate(path, 1):
        cost = transition_cost(x, values, step, () if step == 1 else path[step-2], scale, config)
        total += sum(float(cost[i, j]) for i, j in pairs)
        total += config.unmatched_cost*(len(values[step-1])+len(values[step])-2*len(pairs))
    return float(total)


def _forward(x, values, scale, config):
    states = [(0., ())]; generated = 0; peak = 1
    for step in range(1, len(x)):
        by_last = {}
        for score, path in states:
            cost = transition_cost(x, values, step, path[-1] if path else (), scale, config)
            for pairs, value in matching_proposals(cost, config):
                generated += 1
                proposal = (score+value, path+(pairs,))
                if pairs not in by_last or proposal[0] < by_last[pairs][0]:
                    by_last[pairs] = proposal
        states = sorted(by_last.values(), key=lambda row: (row[0], row[1]))[:config.beam]
        peak = max(peak, len(states))
    return states, {'generated_states': generated, 'peak_states': peak}


def reconstruct_line(x, values, config=StripConfig()):
    x = np.asarray(x, dtype=float)
    if len(x) != len(values) or np.any(np.diff(x) == 0):
        raise ValueError('line coordinates must be distinct at consecutive samples')
    values = tuple(np.asarray(v, dtype=float) for v in values)
    if any(not np.all(np.isfinite(v)) for v in values):
        raise ValueError('candidate scalar values must be finite')
    if not len(x):
        return (), {'objective': 0., 'generated_states': 0, 'peak_states': 0}
    scale = scalar_scale(values)
    forward, fs = _forward(x, values, scale, config)
    reverse, rs = _forward(x[::-1], values[::-1], scale, config)
    competitors = {}
    for direction, states in [('forward', forward), ('reverse', reverse)]:
        for score, path in states:
            if direction == 'reverse':
                path = tuple(tuple(sorted((j, i) for i, j in pairs)) for pairs in path[::-1])
            checked = line_objective(x, values, path, scale, config)
            if not np.isclose(checked, score, rtol=1e-9, atol=1e-9):
                raise AssertionError('forward and reverse objective are not identical')
            competitors.setdefault(path, (checked, direction))
    ranked = sorted(competitors.items(), key=lambda row: (row[1][0], row[0]))
    best, (score, direction) = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    forward_path, reverse_path = forward[0][1], tuple(
        tuple(sorted((j, i) for i, j in pairs)) for pairs in reverse[0][1][::-1])
    return best, {'objective': score, 'scale': scale, 'selected_direction': direction,
        'direction_agreement': forward_path == reverse_path,
        'competing_path': None if runner_up is None else runner_up[0],
        'runner_up_gap': None if runner_up is None else runner_up[1][0]-score,
        'generated_states': fs['generated_states']+rs['generated_states'],
        'peak_states': max(fs['peak_states'], rs['peak_states']),
        'solver': 'bidirectional beam; no claim of global optimality'}


def scalar_faces(c, f, pairs, part):
    """Close faces by projected geometry, including coincident raw candidates."""
    values = tuple(getattr(z, part) for z in f.values)
    tol = 1e-9*max(1., max((float(np.max(abs(z))) for z in values if len(z)), default=1.))
    lookup = c.edge_index(); maps = {}
    for eid, rows in enumerate(pairs):
        a, b = c.edges[eid]
        maps[a, b] = dict(rows); maps[b, a] = {j: i for i, j in rows}
    faces = []; unresolved = []; equivalent = 0
    for fid, sites in enumerate(c.faces):
        for start in range(len(values[sites[0]])):
            paths = [(start,)]
            for a, b in zip(sites, sites[1:]+sites[:1]):
                following = []
                for path in paths:
                    source = path[-1]
                    candidates = [source] if source in maps[a, b] else []
                    candidates += [i for i in maps[a, b] if i != source and abs(values[a][i]-values[a][source]) <= tol]
                    for i in candidates:
                        following.append(path+(maps[a, b][i],))
                paths = sorted(set(following))
            closed = [path for path in paths if abs(values[sites[0]][path[-1]]-values[sites[0]][start]) <= tol]
            if not closed:
                unresolved.append({'face': fid, 'start': [sites[0], start],
                                   'return_candidates': sorted({p[-1] for p in paths})})
                continue
            path = min(closed, key=lambda row: (row[-1] != start, row))
            equivalent += path[-1] != start
            refs = tuple(CandidateRef(s, i) for s, i in zip(sites, path[:-1]))
            faces.append(CoverFace(fid, refs, .5, False))
    return tuple(faces), {'unclosed_cells': unresolved, 'equivalent_scalar_closures': equivalent}


def reconstruct_scalar(c, f, *, part='real', config=StripConfig()):
    if part not in ('real', 'imag'):
        raise ValueError('selected projection must be real or imag')
    if c.grid_shape is None or c.coordinate_dimension not in (1, 2):
        raise ValueError('scalar strips require a regular 1D or 2D parameter grid')
    values = tuple(getattr(z, part) for z in f.values)
    grid = np.arange(c.n_sites).reshape(c.grid_shape); lookup = c.edge_index()
    pairs = [() for _ in c.edges]; rows = []; edges = []
    for axis in range(c.coordinate_dimension):
        for sites in np.moveaxis(grid, axis, -1).reshape(-1, c.grid_shape[axis]):
            # Respect deleted parameter edges; do not bridge across them.
            breaks = [0]+[i for i in range(1, len(sites)) if tuple(sorted(sites[i-1:i+1])) not in lookup]+[len(sites)]
            for lo, hi in zip(breaks, breaks[1:]):
                segment = sites[lo:hi]
                if len(segment) < 2:
                    continue
                x = c.coordinates[segment, axis]
                path, info = reconstruct_line(x, tuple(values[s] for s in segment), config)
                rows.append({'axis': axis, 'sites': segment.tolist(), **info})
                for (a, b), matching in zip(zip(segment, segment[1:]), path):
                    eid = lookup[min(a, b), max(a, b)]
                    canonical = matching if a < b else tuple((j, i) for i, j in matching)
                    pairs[eid] = canonical
                    for i, j in canonical:
                        s, t = c.edges[eid]
                        edges.append(AssociationEdge(eid, CandidateRef(s, i), CandidateRef(t, j),
                            float(abs(values[s][i]-values[t][j])/info['scale']), .5, False, ()))
    refs = tuple(tuple(CandidateRef(s, i) for i in range(len(v))) for s, v in enumerate(values))
    faces, assembly = scalar_faces(c, f, pairs, part)
    components, confidence, edges = _assemble_components(c, f, edges, faces, refs, set(), .95)
    info = {'version': 'scalar-strip-p2', 'part': part, 'config': vars(config),
            'generated_measurements': 0, 'raw_candidates': f.n_candidates,
            'physical_fields_used': False, 'other_frequency_projection_used': False,
            'lines': rows, 'generated_states': sum(r['generated_states'] for r in rows),
            'direction_disagreements': sum(not r['direction_agreement'] for r in rows),
            'stage_qualified': False, **assembly}
    return BandCoverMesh(c, f, tuple(edges), faces, components, confidence,
                         diagnostics={'scalar_reconstruction': info}), info
