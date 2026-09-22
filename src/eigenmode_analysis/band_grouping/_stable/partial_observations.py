"""G18 conditional spectral evidence from incomplete observed root sets.

Latent polynomial roots are internal model values, never delivered samples.
Only coordinates and complex frequencies enter this module; fixed membership
is a condition, not independent end-to-end validation.
"""
from itertools import permutations, product
import numpy as np
from .local_spectral_contact import SpectralModel, fit, predict, contacts, patch_bounds
from .polynomial import design
from .projected_connections import restore_model


def observed_error(roots, observed):
    """Exact finite bottleneck injection for the bounded (<=5) research block."""
    roots, observed = np.asarray(roots), np.asarray(observed)
    if len(roots) > 5 or len(observed) > len(roots):
        raise ValueError('bounded incomplete spectrum requires 0 <= observations <= roots <= 5')
    if not np.isfinite(roots).all() or not np.isfinite(observed).all():
        raise ValueError('finite frequencies required')
    choices = permutations(range(len(roots)), len(observed))
    return min((float(np.max(abs(roots[list(p)]-observed), initial=0.)), tuple(p)) for p in choices)


def fit_observed(xy, spectra, k, *, degree=3, mode='observed'):
    """Monic P(z,x)=0 on measured roots; no completed site spectra are fitted.

    A constant frequency shift and scale use TRAINING observations only.
    The old complete-spectrum coefficient model is an explicit comparison.
    The algebraic LS solution must separately pass root-unit error checks.
    """
    xy = np.asarray(xy, float)
    spectra = tuple(np.asarray(z, complex) for z in spectra)
    if (k not in range(2,6) or degree not in (2,3) or len(xy) != len(spectra)
            or xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all()
            or any(z.ndim != 1 or len(z)>k or not np.isfinite(z).all() for z in spectra)):
        raise ValueError('finite coordinates and bounded observed spectra required')
    if mode == 'observed_fullsites':
        full = np.array([len(z)==k for z in spectra])
        model, info = fit_observed(xy[full], [z for z in spectra if len(z)==k], k, degree=degree, mode='observed')
        return model, {**info,'mode':mode,'partial_training_sites_unused':int((~full).sum())}
    if mode == 'complete':
        full = np.array([len(z)==k for z in spectra])
        model = fit(xy[full], np.array([z for z in spectra if len(z)==k]), degree)
        return model, {'mode':mode,'complete_training_sites':int(full.sum()), 'partial_training_sites_unused':int((~full).sum())}
    if mode != 'observed':
        raise ValueError('unknown partial-spectrum model')
    p = design(xy, degree).shape[1]; variables = p*k
    measured = np.concatenate(spectra) if spectra else np.array([],complex)
    if len(measured) < 2*variables:
        raise ValueError('insufficient observed-root equations for fixed polynomial order')
    center = complex(np.mean(measured))
    scale = max(float(np.max(abs(measured-center))), 1e-12)
    basis = design(xy, degree)
    rows, target = [], []
    for a, zs in zip(basis, spectra):
        for z in (zs-center)/scale:
            rows.append((a[:,None]*z**np.arange(k-1,-1,-1)).ravel())
            target.append(-z**k)
    matrix = np.asarray(rows); target = np.asarray(target)
    coef, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=1e-12)
    if rank != variables:
        raise ValueError('rank deficient observed-root equations; missing spectrum is not identified')
    condition = float(singular[0]/singular[-1])
    mean = np.zeros(p, complex); mean[0] = center
    model = SpectralModel(degree, mean, coef.reshape(p,k), scale)
    info = {'mode':mode,'equations':len(rows),'variables':variables,'rank':int(rank),
        'design_condition_number':condition,'rank_rcond':1e-12,
        'algebraic_residual_max':float(np.max(abs(matrix@coef-target))),
        'complete_training_sites':sum(len(z)==k for z in spectra),
        'partial_training_sites_used':sum(0<len(z)<k for z in spectra)}
    return model, info


def diagnose_partial(c, values, members, sites, k, *, mode='observed', epsilon=5e-5, fit_function=None):
    lo, hi = patch_bounds(c, sites, 'shifted')
    patch = [c.site_index((i,j)) for i in range(lo[0],hi[0]) for j in range(lo[1],hi[1])]
    box = c.coordinates[patch]
    center, scale = (box.min(0)+box.max(0))/2, np.ptp(box,axis=0)/2
    row = {'sites':list(sites),'all_patch_sites':patch,'members':[list(members[s]) for s in patch],
        'k':k,'degree':3,'mode':mode,'epsilon_f':epsilon,'status':'unsupported',
        'withheld_sites':list(sites),'latent_roots_are_measurements':False,
        'membership_condition':'Frozen unordered membership used full data; NOT independent end-to-end holdout.'}
    if np.any(scale==0):
        return {**row,'reason':'not a two-dimensional native patch'}
    xy = (box-center)/scale
    spectra = [np.asarray(values[s])[members[s]] for s in patch]
    held = [patch.index(s) for s in sites]
    train = [i for i in range(len(patch)) if i not in held]
    def fitting(indices):
        # Alternative research fitters receive TRAINING values only. The G18
        # default remains byte-for-byte the same numerical fitting function.
        fitter = fit_observed if fit_function is None else fit_function
        return fitter(xy[indices], [spectra[i] for i in indices], k, mode=mode)
    try:
        model, info = fitting(train)
        pred = predict(model,xy)
        errors = [observed_error(p,z)[0] for p,z in zip(pred,spectra)]
        loo = []
        for i in train:
            m, check = fitting([j for j in train if j!=i])
            error = observed_error(predict(m,xy[[i]])[0],spectra[i])[0]
            loo.append({'site':patch[i],'error':error,'fit_condition':check.get('design_condition_number')})
        row.update(training_max=max(errors[i] for i in train),held_max=max(errors[i] for i in held),
            training_loo_max=max(r['error'] for r in loo),training_loo=loo,fit_info=info,
            model={'center':center.tolist(),'coordinate_scale':scale.tolist(),'frequency_scale':model.scale,
                   'mean':[[z.real,z.imag] for z in model.mean],
                   'coefficients':[[[z.real,z.imag] for z in a] for a in model.coefficients]})
        row['needed_epsilon_f'] = max(row['training_max'],row['held_max'],row['training_loo_max'])
        row['status'] = 'supported_conditional' if row['needed_epsilon_f']<=epsilon else 'outside_declared_tolerance'
        if row['status']=='supported_conditional':
            # Name ALL latent roots independently at the two endpoints. Never
            # index a missing observation with -1 or supply it a pseudo-value.
            endpoints = np.array([np.array(sorted(pred[i],key=lambda z:(z.real,z.imag))) for i in held])
            row['latent_endpoints'] = [[[z.real,z.imag] for z in p] for p in endpoints]
            row['continuation'] = contacts(model,xy[held],endpoints)
        return row
    except (ValueError, np.linalg.LinAlgError) as error:
        return {**row,'status':'unsupported','reason':str(error)}


def partial_family(continuation, left_roots, right_roots, left_observed, right_observed,
                   *, epsilon=5e-5, max_options=64):
    """Enumerate observed links, with absent roots NEVER made into nodes.

    Retain root-contact histories because a partial observed path needs its
    actual intermediate branch history, unlike a complete unordered set.
    """
    from .projected_connections import contact_options
    allowed, reason = contact_options(continuation,max_options=max_options)
    if allowed is None:return None,reason
    k=len(left_roots)
    def injections(roots, z):
        return [p for p in permutations(range(k),len(z))
                if max((abs(z[i]-roots[j]) for i,j in enumerate(p)),default=0.)<=epsilon]
    left=injections(left_roots,left_observed);right=injections(right_roots,right_observed)
    if not left or not right:return None,'no_endpoint_injection_within_declared_tolerance'
    events=sorted(continuation['contacts'],key=lambda x:x['t'])
    if 2**len(events)*len(left)*len(right)>4096:return None,'partial_history_budget_exceeded'
    terminal=continuation['fine']['endpoint_mapping'];histories=[]
    for bits in product((0,1),repeat=len(events)):
        state=list(range(k));states=[]
        for event,bit in zip(events,bits):
            if bit:
                j,l=event['left_local_columns'];state=[l if v==j else j if v==l else v for v in state]
            states.append(state.copy())
        full=tuple(terminal[v] for v in state)
        if full not in allowed:raise RuntimeError('contact history inconsistent with G17 family')
        for a,b in product(left,right):
            lookup={root:i for i,root in enumerate(b)}
            pairs=tuple((i,lookup[full[root]]) for i,root in enumerate(a) if full[root] in lookup)
            histories.append({'pairs':pairs,'left_injection':a,'right_injection':b,
                              'states':states,'switches':bits,'terminal_mapping':full})
    if len({h['pairs'] for h in histories})>max_options:return None,'partial_family_budget_exceeded'
    return histories,None


def baseline_pairs(baseline, bands, members, sites):
    s,t=sites;ls={int(raw):int(b) for b,raw in enumerate(baseline[s,bands]) if raw>=0}
    rs={int(raw):int(b) for b,raw in enumerate(baseline[t,bands]) if raw>=0}
    return tuple((i,j) for i,raw in enumerate(members[s]) for j,other in enumerate(members[t]) if ls[raw]==rs[other])


def partial_factor(c, values, members, baseline, bands, sites, record, *, epsilon=5e-5):
    s,t=sites;k=len(bands);old=baseline_pairs(baseline,bands,members,sites)
    result={'sites':list(sites),'sizes':[len(members[s]),len(members[t])],
        'status':record['status'],'supported':False,'frozen_fallback':True,
        'options':[{'pairs':[list(p) for p in old],'cost':0.}],
        'baseline_pairs':[list(p) for p in old], 'needed_epsilon_f':record.get('needed_epsilon_f')}
    if record['status']!='supported_conditional':return result
    roots=np.asarray(record['latent_endpoints']);roots=roots[...,0]+1j*roots[...,1]
    left,right=values[s][members[s]],values[t][members[t]]
    histories,reason=partial_family(record['continuation'],roots[0],roots[1],left,right,epsilon=epsilon)
    if histories is None:return {**result,'reason':reason}
    from .local_spectral_contact import trace
    model=restore_model(record);saved=record['model']
    ends=(c.coordinates[[s,t]]-saved['center'])/saved['coordinate_scale']
    _,times,tracked=trace(model,ends,roots,steps=128)
    events=sorted(record['continuation']['contacts'],key=lambda x:x['t'])
    options={}
    for h in histories:
        # For complete endpoints this is exactly the unordered full model set.
        # For partial endpoints compare only witnessed paths with real data at
        # BOTH ends. Record match counts; absence never earns fictitious fit.
        ids=[p[0] for p in h['pairs']];js=[p[1] for p in h['pairs']]
        path_roots=np.tile(np.arange(k),(len(times),1))
        for event,state in zip(events,h['states']):path_roots[times>=event['t']]=state
        if ids:
            columns=path_roots[:,np.array(h['left_injection'])[ids]]
            target=np.sort(np.take_along_axis(tracked,columns,axis=1).real,axis=1)
            actual=np.sort((1-times[:,None])*left[ids].real+times[:,None]*right[js].real,axis=1)
            cost=float(np.max(abs(actual-target)))
        else:cost=0.
        pairs=h['pairs'];entry={'pairs':[list(p) for p in pairs],'cost':cost,'matched':len(pairs),
            'left_unmatched':[i for i in range(len(left)) if i not in ids],
            'right_unmatched':[j for j in range(len(right)) if j not in js],
            'witness':h}
        if pairs not in options or cost<options[pairs]['cost']:options[pairs]=entry
    return {**result,'supported':True,'frozen_fallback':False,'options':[options[p] for p in sorted(options)],
        'contacts':len(events),'model_endpoint_injection_count':len(histories),
        'cost_scope':'finite 128-step error of observed endpoint chords against witnessed model-Re paths; not physical error',
        'admitted_match_counts':sorted({len(p) for p in options})}
