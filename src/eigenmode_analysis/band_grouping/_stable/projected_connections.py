"""Numerical subset of frozen Stable v1; see migration provenance."""
import numpy as np
from .local_spectral_contact import SpectralModel, predict


def restore_model(record):
    saved = record['model']
    def z(a):
        a = np.asarray(a, float)
        return a[..., 0] + 1j*a[..., 1]
    return SpectralModel(record['degree'], z(saved['mean']),
                         z(saved['coefficients']), saved['frequency_scale'])


def contact_options(continuation, *, max_options=64):
    """Enumerate choices at ordered isolated Re contacts, not arbitrary swaps.

    Output maps a left baseline-column to a right baseline-column. Tracked
    model roots are named at the left; switches exchange which root a grouped
    real path follows. Simultaneous contacts sharing a root are unsupported.
    """
    if not continuation.get('stable_numerical_continuation'):
        return None, 'unresolved_complex_root_trace'
    terminal = tuple(continuation['fine']['endpoint_mapping'])
    k = len(terminal)
    if sorted(terminal) != list(range(k)):
        raise ValueError('terminal model-root map is not a permutation')
    events = sorted(continuation['contacts'], key=lambda e: e['t'])
    for e in events:
        pair = e['left_local_columns']
        if len(pair) != 2 or pair[0] == pair[1] or any(j < 0 or j >= k for j in pair):
            raise ValueError('invalid model contact root indices')
        if not np.isfinite(e['t']) or not 0 <= e['t'] <= 1:
            raise ValueError('contact lies outside the original edge')
    for a, b in zip(events, events[1:]):
        if abs(a['t']-b['t']) < 1e-9 and set(a['left_local_columns']) & set(b['left_local_columns']):
            return None, 'simultaneous_multiroot_contact'
    states = {tuple(range(k))}
    for e in events:
        j, l = e['left_local_columns']
        switched = {tuple(l if v == j else j if v == l else v for v in p) for p in states}
        states |= switched
        if len(states) > max_options:
            return None, 'contact_family_budget_exceeded'
    options = sorted({tuple(terminal[v] for v in state) for state in states})
    return options, None


def raw_chord_costs(c, values, baseline, bands, sites, record, mappings, steps=128):
    """Finite sup Re-set distance, INCLUDING all detected model contacts.

    This is a model-conditional sampled drawing objective, not an analytic
    maximum or a measurement-error certificate. Internal roots are not output.
    """
    s, t = sites
    saved = record['model']; model = restore_model(record)
    times = np.unique(np.r_[np.linspace(0, 1, steps+1),
                            [e['t'] for e in record['continuation']['contacts']]])
    xy = (1-times[:, None])*c.coordinates[s] + times[:, None]*c.coordinates[t]
    xy = (xy-np.asarray(saved['center']))/np.asarray(saved['coordinate_scale'])
    expected = np.sort(predict(model, xy).real, axis=1)
    left = values[s][baseline[s, bands]].real
    right = values[t][baseline[t, bands]].real
    costs = []
    for p in mappings:
        actual = np.sort((1-times[:, None])*left + times[:, None]*right[np.asarray(p)], axis=1)
        costs.append(float(np.max(abs(actual-expected))))
    return costs


def evidence_edge(c, values, baseline, bands, sites, record, *, epsilon=5e-5, protect_baseline=True):
    """One finite connection factor; lack of support freezes the old mapping."""
    identity = tuple(range(len(bands)))
    row = {'sites': list(sites), 'status': record['status'], 'model_edge': record.get('edge'),
           'options': [{'mapping': list(identity), 'cost': 0.}], 'supported': False,
           'frozen_fallback': True, 'regression_guard': protect_baseline, 'needed_epsilon_f': record.get('needed_epsilon_f')}
    if record['status'] != 'supported_conditional':
        return row
    mappings, reason = contact_options(record.get('continuation', {}))
    if mappings is None:
        row['reason'] = reason
        return row
    identity_cost = raw_chord_costs(c, values, baseline, bands, sites, record, [identity])[0]
    costs = raw_chord_costs(c, values, baseline, bands, sites, record, mappings)
    options = [{'mapping': list(p), 'cost': cost} for p, cost in zip(mappings, costs)
               if not protect_baseline or cost <= identity_cost+epsilon]
    # An empty supported factor is a real conditional infeasibility. Do not
    # turn it silently into an unsupported factor to force an output success.
    row.update(options=options, supported=True, frozen_fallback=False,
               baseline_model_cost=identity_cost, admissible_before_cost_guard=len(mappings),
               rejected_by_local_regression_guard=len(mappings)-len(options),
               contacts=len(record['continuation']['contacts']))
    return row
