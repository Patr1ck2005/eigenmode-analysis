"""Research G2: use checked scalar models to label original samples only.

The existing model families are hypotheses fitted to Re(f), not measured
complex eigenvalues. Internal continuation points never become output data.
Nontrivial continuation cycles reject the whole model for ordinary band IDs;
this does not imply that its projected scalar set cannot be drawn otherwise.
"""
import numpy as np
from numpy.polynomial import Polynomial

from .model import AssociationEdge, BandCoverMesh, CandidateRef, CoverFace
from .scalar_models import basis, fit_collection, powers
from .tracking import _assemble_components


def _branches(model, xy):
    p = (np.asarray(xy)-model['xy_center'])/model['xy_radius']
    mean = basis(p, 2)@model['mean']
    kind = model['kind']
    if kind == 'single':
        return mean[:, None], None
    v = basis(p, model['seam_degree'])@model['seam']
    if kind == 'pair-polynomial':
        return mean[:, None]+v[:, None]*np.array([-1., 1.]), None
    u = basis(p, model['u_degree'])@model['u']
    order = 2 if kind == 'pair-root' else 3
    w = u+1j*v*(2 if order == 2 else 1)
    angle = np.unwrap(np.angle(w))
    branches = mean[:, None]+abs(w[:, None])**(1/order)*np.cos(
        angle[:, None]/order+2*np.pi*np.arange(order)/order)
    return branches, w


def _restriction(model, a, b, name):
    a, b = (np.asarray([a, b])-model['xy_center'])/model['xy_radius']
    x, y = Polynomial([a[0], b[0]-a[0]]), Polynomial([a[1], b[1]-a[1]])
    return sum((float(v)*x**i*y**j for (i, j), v in
                zip(powers(model[name+'_degree']), model[name])), Polynomial([0.]))


def _phase_increment(polynomial):
    """Continuous phase of a low-degree polynomial on t in [0,1].

    Each factor (t-root) traces a straight segment without a zero. Its
    principal endpoint phase difference is its continuous phase change;
    summing factors retains turns a fixed sampling grid can miss.
    This is floating-point model algebra, not a certified numerical bound.
    """
    coefficients = polynomial.coef
    scale = float(np.max(abs(coefficients)))
    if scale == 0:
        raise ValueError('modeled branch collision on edge')
    normalized = Polynomial(coefficients/scale).trim(tol=1e-14)
    roots = normalized.roots()
    if not np.all(np.isfinite(roots)):
        raise ValueError('nonfinite polynomial roots')
    if any(-1e-10 <= r.real <= 1+1e-10 and abs(r.imag) <= 1e-10 for r in roots):
        raise ValueError('modeled branch collision or unresolved near-real polynomial root')
    probes = np.linspace(0., 1., 9)
    reconstructed = normalized.coef[-1]*np.prod(probes[:, None]-roots[None, :], axis=1)
    if np.max(abs(reconstructed-normalized(probes))) > 1e-10:
        raise ValueError('polynomial root factorization did not reproduce the model')
    rebuilt = Polynomial.fromroots(roots).coef*normalized.coef[-1]
    padded = np.pad(rebuilt, (0, len(coefficients)-len(rebuilt)))
    if np.max(abs(padded-coefficients/scale)) > 1e-10:
        raise ValueError('polynomial coefficient reconstruction is unresolved')
    if min(abs(normalized(0.)), abs(normalized(1.))) <= 1e-12:
        raise ValueError('modeled branch collision near an endpoint')
    start, end = -roots, 1-roots
    delta = float(np.sum(np.angle((end/abs(end))*np.conj(start/abs(start)))))
    endpoint_phase = np.angle(normalized(1.))-np.angle(normalized(0.))
    if abs(np.angle(np.exp(1j*(delta-endpoint_phase)))) > 1e-8:
        raise ValueError('polynomial phase disagrees with its endpoints')
    return delta


def _edge_permutation(model, a, b):
    """Transport fitted branches by model algebra, returning raw scalar ranks."""
    xy = np.asarray([a, b])
    values, w = _branches(model, xy)
    if w is not None:
        order = 2 if model['kind'] == 'pair-root' else 3
        polynomial = _restriction(model, a, b, 'u')+1j*(2 if order == 2 else 1)*_restriction(model, a, b, 'seam')
        try:
            delta = _phase_increment(polynomial)
        except (ValueError, np.linalg.LinAlgError) as exception:
            return None, str(exception)
        p = (xy-model['xy_center'])/model['xy_radius']
        mean = basis(p, 2)@model['mean']
        phase = np.array([np.angle(w[0]), np.angle(w[0])+delta])
        values = mean[:, None]+abs(w[:, None])**(1/order)*np.cos(
            phase[:, None]/order+2*np.pi*np.arange(order)/order)
    left, right = np.argsort(values, axis=1, kind='stable')
    if values.shape[1] > 1 and np.min(np.diff(np.sort(values, axis=1), axis=1)) < 1e-10:
        return None, 'coincident endpoint ranks need an explicit scalar convention'
    return tuple(int(np.flatnonzero(right == column)[0]) for column in left), None


def _components(refs, pairs):
    parent = {r: r for r in refs}
    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r
    for a, b in pairs:
        parent[find(b)] = find(a)
    groups = {}
    for r in sorted(refs):
        groups.setdefault(find(r), []).append(r)
    result = list(groups.values())
    if any(len({r.site for r in row}) != len(row) for row in result):
        raise ValueError('model continuation has a same-site cycle conflict')
    return result


def _mesh(c, f, groups, info):
    refs = tuple(tuple(CandidateRef(s, j) for j in range(len(z))) for s, z in enumerate(f.values))
    label = {r: k for k, group in enumerate(groups) for r in group}
    if len(label) != f.n_candidates or sum(map(len, groups)) != f.n_candidates:
        raise ValueError('grouping must preserve each original candidate exactly once')
    by_site = [{label[r]: r for r in row} for row in refs]
    edges = []
    for eid, (a, b) in enumerate(c.edges):
        for k in sorted(by_site[a].keys() & by_site[b].keys()):
            edges.append(AssociationEdge(eid, by_site[a][k], by_site[b][k], 0., .5, False, ()))
    faces = []
    for fid, sites in enumerate(c.faces):
        for k in sorted(set.intersection(*(set(by_site[s]) for s in sites))):
            faces.append(CoverFace(fid, tuple(by_site[s][k] for s in sites), .5, False))
    components, confidence, edges = _assemble_components(c, f, edges, faces, refs, set(), .95)
    return BandCoverMesh(c, f, tuple(edges), tuple(faces), components, confidence,
                         diagnostics={'scalar_model_grouping': info})


def reconstruct_model_grouping(c, f, *, fallback=None):
    """Reuse model support, change only sample membership, retain G1 otherwise.

    No generator, original row ID, measured Im(f), or physical field selects
    associations. This model family is not a universal reconstruction claim.
    """
    if fallback is None:
        raise ValueError("internal scalar grouping requires the precomputed transport seed")
    if fallback.complex is not c or fallback.fibers is not f:
        raise ValueError('fallback must reference the same original input objects')
    models, fitted = fit_collection(c, f, 'real')
    groups, claimed, attempts = [], set(), []
    for model in models:
        refs = np.asarray(model['references'], dtype=int)
        known = {CandidateRef(s, int(j)) for s, row in enumerate(refs) for j in row if j >= 0}
        pairs, latent_pairs, error = [], [], None
        # A fitting residual must not turn equal measured candidates into
        # apparently distinct endpoint ranks. Keep their multiplicity, but
        # leave this model's ownership decision to the declared fallback.
        for s, row in enumerate(refs):
            raw = f.values[s].real[row[row >= 0]]
            tolerance = max(1e-10*model.get('height_scale', 1.),
                            64*np.finfo(float).eps*max(1., float(np.max(abs(raw), initial=0.))))
            if len(raw) > 1 and np.min(np.diff(np.sort(raw))) <= tolerance:
                error = 'coincident raw endpoint ranks need an explicit scalar convention'
                break
        for a, b in c.edges:
            if error:
                break
            permutation, error = _edge_permutation(model, c.coordinates[a], c.coordinates[b])
            if error:
                break
            for left, right in enumerate(permutation):
                # Check the complete inferred rank graph before dropping
                # absent observations. A missing branch must not hide a
                # nontrivial cycle that ordinary grouped plotting re-closes.
                latent_pairs.append((CandidateRef(a, left), CandidateRef(b, right)))
                i, j = refs[a, left], refs[b, right]
                if i >= 0 and j >= 0:
                    pairs.append((CandidateRef(a, int(i)), CandidateRef(b, int(j))))
        if error is None:
            try:
                _components({CandidateRef(s, rank) for s in range(c.n_sites)
                             for rank in range(refs.shape[1])}, latent_pairs)
                candidate_groups = _components(known, pairs)
            except ValueError as exception:
                error = str(exception)
        if known & claimed:
            error = 'overlapping checked model ownership'
        accepted = error is None
        attempts.append({'kind': model['kind'], 'accepted': accepted, 'reason': error,
                         'raw_candidates': len(known), 'native_relative_error': model['native_relative_error'],
                         'held_out_relative_errors': model['held_out_relative_errors'], 'model': model})
        if accepted:
            groups.extend(candidate_groups)
            claimed.update(known)
    remaining = set(f.refs())-claimed
    by_site = [[r for r in row if r in remaining] for row in
               (tuple(CandidateRef(s, j) for j in range(len(z))) for s, z in enumerate(f.values))]
    unique_remainder = max(map(len, by_site), default=0) <= 1
    # Reuse P9's single remaining sheet convention, without spanning absent
    # sites. Old mixed fallback labels must not split this observed sheet.
    if unique_remainder:
        remainder_pairs = [(by_site[a][0], by_site[b][0]) for a, b in c.edges
                           if by_site[a] and by_site[b]]
    else:
        remainder_pairs = [(e.source, e.target) for e in fallback.edges
                           if e.source in remaining and e.target in remaining]
    groups.extend(_components(remaining, remainder_pairs))
    info = {'version': 'scalar-model-grouping-g2', 'models': attempts, 'fit_report': fitted,
            'model_candidates_grouped': len(claimed), 'fallback_candidates': len(remaining),
            'single_remainder_adjacency': unique_remainder,
            'accepted_models': sum(a['accepted'] for a in attempts),
            'input_policy': 'coordinates and Re(f) only; latent complex fields are fitted hypotheses',
            'generated_measurements': 0, 'derived_vertices': 0, 'stage_qualified': False,
            'physical_fields_used': False, 'other_frequency_projection_used': False,
            'production_default': False,
            'limitation': 'Checked model support and floating-point polynomial continuation are not a uniqueness or continuous error certificate'}
    return _mesh(c, f, groups, info), info
