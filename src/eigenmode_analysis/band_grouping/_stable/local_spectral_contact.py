"""Conditional 2D spectral-model evidence, NOT a replacement classifier.

Reuses the mean-centered polynomial representation from R20. The new question
is whether G15's disputed pair is a closed spectral subsystem. No payload,
reference labels, imputed samples, or model roots enter delivered band rows.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment, brentq
from .polynomial import design


@dataclass
class SpectralModel:
    degree: int
    mean: np.ndarray
    coefficients: np.ndarray
    scale: float


def fit(xy, z, degree=3):
    xy, z = np.asarray(xy, float), np.asarray(z, complex)
    if degree not in (2, 3):
        raise ValueError('only the declared quadratic/cubic numerical families are supported')
    if (xy.ndim != 2 or xy.shape[1] != 2 or z.ndim != 2 or len(z) != len(xy)
            or z.shape[1] < 2 or not np.isfinite(xy).all() or not np.isfinite(z).all()):
        raise ValueError('finite 2D coordinates and complete unordered spectra required')
    a = design(xy, degree)
    if len(a) < 2*a.shape[1] or np.linalg.matrix_rank(a) != a.shape[1]:
        raise ValueError('insufficient independent samples for the declared polynomial family')
    mu = z.mean(axis=1)
    scale = max(float(np.max(abs(z-mu[:, None]))), 1e-12)
    polynomial = np.array([np.poly((v-m)/scale)[1:] for v, m in zip(z, mu)])
    return SpectralModel(degree, np.linalg.lstsq(a, mu, rcond=None)[0],
        np.linalg.lstsq(a, polynomial, rcond=None)[0], scale)


def predict(model, xy):
    a = design(np.asarray(xy, float), model.degree)
    return np.array([np.roots(np.r_[1., coefficients])*model.scale+mu
        for coefficients, mu in zip(a@model.coefficients, a@model.mean)])


def compare(predicted, measured):
    cost = abs(np.asarray(predicted)[:, None]-np.asarray(measured)[None, :])
    rows, columns = linear_sum_assignment(cost)
    return columns, float(np.max(cost[rows, columns]))


def fit_withheld(xy, z, withheld, degree=3):
    """Values at withheld positions cannot influence coefficients or scale."""
    xy, z = np.asarray(xy, float), np.asarray(z, complex)
    train = np.ones(len(xy), bool)
    train[np.asarray(withheld, int)] = False
    return fit(xy[train], z[train], degree), train


def trace(model, endpoints, raw_endpoints, steps=128):
    """Numerical root continuation only; collisions return unresolved.

    Internal samples never become output data. Endpoint row names are fixed
    by a one-to-one match at the left endpoint, not by scalar sorting.
    """
    endpoints = np.asarray(endpoints, float)
    t = np.linspace(0., 1., steps+1)
    roots = predict(model, (1-t[:, None])*endpoints[0]+t[:, None]*endpoints[1])
    mapping, _ = compare(roots[0], raw_endpoints[0])
    roots[0] = roots[0, np.argsort(mapping)]
    min_gap, ratio = np.inf, 0.
    for i in range(len(roots)):
        gaps = [abs(roots[i, j]-roots[i, k]) for j, k in combinations(range(roots.shape[1]), 2)]
        gap = float(min(gaps))
        min_gap = min(min_gap, gap)
        if i:
            p, _ = compare(roots[i-1], roots[i])
            roots[i] = roots[i, p]
            previous_gap = min(abs(roots[i-1, j]-roots[i-1, k]) for j, k in combinations(range(roots.shape[1]), 2))
            ratio = max(ratio, float(np.max(abs(roots[i]-roots[i-1]))/max(min(gap, previous_gap), 1e-15)))
    p, endpoint_error = compare(roots[-1], raw_endpoints[1])
    numerical_floor = 1e-8*max(model.scale, 1e-12)
    return {'endpoint_mapping': p.tolist(), 'minimum_complex_gap': min_gap,
        'maximum_step_to_gap_ratio': ratio, 'endpoint_error': endpoint_error,
        'resolved': bool(min_gap > numerical_floor and ratio < .25)}, t, roots


def contacts(model, endpoints, raw_endpoints):
    coarse, _, _ = trace(model, endpoints, raw_endpoints, steps=64)
    fine, t, roots = trace(model, endpoints, raw_endpoints, steps=128)
    stable = (coarse['resolved'] and fine['resolved'] and
              coarse['endpoint_mapping'] == fine['endpoint_mapping'])
    result = {'coarse': coarse, 'fine': fine, 'stable_numerical_continuation': stable,
        'contacts': [], 'scope': 'contacts of the fitted model only, not physical contact certification'}
    if not stable:
        result['status'] = 'unresolved_numerical_root_continuation'
        return result
    for j, k in combinations(range(roots.shape[1]), 2):
        delta = (roots[:, j]-roots[:, k]).real
        found = []
        for i in range(len(t)-1):
            if delta[i]*delta[i+1] > 0:
                continue
            if delta[i] == delta[i+1] == 0:
                continue  # Extended projection equality is not an isolated crossing.
            def value(q, full=False):
                xy = (1-q)*endpoints[0]+q*endpoints[1]
                r = predict(model, np.asarray([xy]))[0]
                alpha = (q-t[i])/(t[i+1]-t[i])
                target = (1-alpha)*roots[i]+alpha*roots[i+1]
                p, _ = compare(target, r)
                r = r[p]
                return r if full else float((r[j]-r[k]).real)
            q = float(brentq(value, t[i], t[i+1], xtol=1e-13))
            if any(abs(q-old) < 1e-9 for old in found):
                continue
            found.append(q)
            r = value(q, full=True)
            result['contacts'].append({'left_local_columns': [j, k], 't': q,
                'normalized_coordinate': ((1-q)*endpoints[0]+q*endpoints[1]).tolist(),
                'real_frequency': float((r[j].real+r[k].real)/2),
                'imaginary_separation': float(abs(r[j].imag-r[k].imag)),
                'real_residual': float(abs(r[j].real-r[k].real))})
    result['status'] = 'conditional_model_contacts' if result['contacts'] else 'no_model_contact_detected'
    return result


def patch_bounds(c, sites, boundary='clipped'):
    """Fixed two-cell halo; optionally translate rather than shrink at a rim.

    Translation uses only original grid coordinates, never data or errors.
    The historical clipped policy remains the default for G16 reproduction.
    """
    if boundary not in ('clipped', 'shifted'):
        raise ValueError('unknown boundary stencil policy')
    ij = np.array([c.grid_index(s) for s in sites])
    shape = np.asarray(c.grid_shape)
    if boundary == 'clipped':
        return np.maximum(ij.min(0)-2, 0), np.minimum(ij.max(0)+3, shape)
    width = np.minimum(ij.max(0)-ij.min(0)+5, shape)
    lo = np.minimum(np.maximum(ij.min(0)-2, 0), shape-width)
    return lo, lo+width


def diagnose(c, values, assignment, bands, edge, *, degree=3, epsilon=5e-5, boundary='clipped'):
    left, right = c.edges[edge]
    lo, hi = patch_bounds(c, (left, right), boundary)
    all_sites = np.array([c.site_index((i, j)) for i in range(lo[0], hi[0]) for j in range(lo[1], hi[1])])
    complete = np.all(assignment[all_sites][:, bands] >= 0, axis=1)
    sites = all_sites[complete]
    row = {'edge': edge, 'bands': list(bands), 'degree': degree, 'epsilon_f': epsilon,
        'all_patch_sites': all_sites.tolist(), 'complete_sites': sites.tolist(),
        'incomplete_sites': all_sites[~complete].tolist(), 'withheld_sites': [left, right],
        'status': 'unsupported', 'qualification': False, 'boundary_policy': boundary,
        'membership_condition': 'Frozen grouping chooses unordered members, not their order. Membership used full data, so this is NOT a fully independent holdout.'}
    if not all(v in sites for v in (left, right)):
        row['reason'] = 'genuine missing endpoint member; no padding'
        return row
    box = c.coordinates[all_sites]
    center, scale = (box.min(0)+box.max(0))/2, np.ptp(box, axis=0)/2
    if np.any(scale == 0):
        row['reason'] = 'not a two-dimensional patch'
        return row
    u = (c.coordinates[sites]-center)/scale
    z = np.array([values[s][assignment[s, bands]] for s in sites])
    held = np.array([int(np.flatnonzero(sites == v)[0]) for v in (left, right)])
    try:
        model, train = fit_withheld(u, z, held, degree)
        pred = predict(model, u)
        errors = np.array([compare(p, v)[1] for p, v in zip(pred, z)])
        # The target edge remains entirely hidden during these training-set LOO fits.
        loo = []
        for i in np.flatnonzero(train):
            m, _ = fit_withheld(u, z, np.r_[held, i], degree)
            loo.append({'site': int(sites[i]), 'max_complex_error': compare(predict(m, u[[i]])[0], z[i])[1]})
        row.update(training_max=float(errors[train].max()), held_max=float(errors[held].max()),
            training_loo_max=max(x['max_complex_error'] for x in loo), training_loo=loo,
            predicted_endpoints=[[[v.real, v.imag] for v in p] for p in pred[held]],
            model={'center': center.tolist(), 'coordinate_scale': scale.tolist(),
                   'frequency_scale': model.scale, 'mean': [[v.real, v.imag] for v in model.mean],
                   'coefficients': [[[v.real, v.imag] for v in p] for p in model.coefficients]})
        needed = max(row['training_max'], row['held_max'], row['training_loo_max'])
        row['needed_epsilon_f'] = needed
        row['status'] = 'supported_conditional' if needed <= epsilon else 'outside_declared_tolerance'
        if row['status'] == 'supported_conditional':
            row['continuation'] = contacts(model, u[held], z[held])
            for item in row['continuation']['contacts']:
                item['original_coordinate'] = (np.array(item['normalized_coordinate'])*scale+center).tolist()
            source_ids = assignment[left, bands]
            end_ids = assignment[right, bands]
            p = row['continuation']['fine']['endpoint_mapping']
            row['proposed_original_links'] = [[left, int(i), right, int(end_ids[j])] for i, j in zip(source_ids, p)]
            # This is an evidence record, never a label mutation or promotion.
    except (ValueError, np.linalg.LinAlgError) as error:
        row['status'] = 'unsupported'
        row['reason'] = str(error)
    return row
