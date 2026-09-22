"""Numerical regressions ported from the SHA-pinned Stable v1 release."""
import itertools
import numpy as np
import pytest
from eigenmode_analysis.band_grouping._stable.partial_observations import fit_observed, observed_error, partial_family, diagnose_partial
from eigenmode_analysis.band_grouping._stable.local_spectral_contact import predict
from eigenmode_analysis.band_grouping._stable.partial_observation_grouping import solve_partial, partial_metrics, actual_pairs


def spectra():
    xy=np.array([(x,y) for x in np.linspace(-1,1,6) for y in np.linspace(-1,1,5)])
    zs=[]
    for i,(x,y) in enumerate(xy):
        h=np.array([[.4+.04*x+.01j,.006,.002*y],[.006,.46-.03*x+.02j,.009],[.002*y,.009,.52+.02*y+.005j]])
        z=np.linalg.eigvals(h);z=z[np.argsort(z.real)]
        zs.append(np.delete(z,i%3) if i%3==0 else z)
    return xy,zs


def test_missing_roots_can_identify_polynomial_without_imputation():
    xy,zs=spectra();model,info=fit_observed(xy,zs,3)
    assert info['partial_training_sites_used']==10 and info['rank']==30
    assert max(observed_error(p,z)[0] for p,z in zip(predict(model,xy),zs))<1e-10
    assert sum(map(len,zs))==80


def test_withheld_values_cannot_change_training_fit_or_scale():
    xy,zs=spectra();keep=list(range(2,len(zs)))
    a,_=fit_observed(xy[keep],[zs[i] for i in keep],3)
    zs[0]=zs[0]+20j;zs[1]=zs[1]+30
    b,_=fit_observed(xy[keep],[zs[i] for i in keep],3)
    np.testing.assert_array_equal(a.mean,b.mean);np.testing.assert_array_equal(a.coefficients,b.coefficients)
    assert a.scale==b.scale


def test_observed_root_order_does_not_classify():
    xy,zs=spectra();a,_=fit_observed(xy,zs,3);b,_=fit_observed(xy,[z[::-1] for z in zs],3)
    assert max(observed_error(p,z)[0] for p,z in zip(predict(a,xy),predict(b,xy)))<1e-10


def test_permanently_missing_branch_is_not_invented_by_rank_deficient_fit():
    xy=np.array([(x,y) for x in np.linspace(-1,1,7) for y in np.linspace(-1,1,7)])
    zs=[np.array([1+.01*x+.03j,2+.02*y+.02j]) for x,y in xy]
    with pytest.raises(ValueError,match='rank deficient'):fit_observed(xy,zs,3)
    with pytest.raises(ValueError):fit_observed(xy[:4],zs[:4],3)


def ct(mapping,contacts=()):
    return {'stable_numerical_continuation':True,'fine':{'endpoint_mapping':mapping},
            'contacts':[{'t':t,'left_local_columns':list(p)} for t,p in contacts]}


def test_equal_observed_counts_do_not_force_all_candidates_to_match():
    roots=np.array([1+.01j,2+.02j,3+.03j])
    h,reason=partial_family(ct([0,1,2]),roots,roots,roots[:2],roots[1:])
    assert reason is None and {v['pairs'] for v in h}=={((1,0),)}
    assert observed_error(roots,roots[:2])[0]==0


def test_coincident_observations_keep_both_endpoint_injections_and_multiplicity():
    roots=np.array([1+.01j,1+.01j,3+.02j])
    h,_=partial_family(ct([0,1,2]),roots,roots,roots[:2],roots[:2])
    assert {v['pairs'] for v in h}=={((0,0),(1,1)),((0,1),(1,0))}


def edge(s,t,options,costs):
    return {'sites':[s,t],'options':[{'pairs':p,'cost':c} for p,c in zip(options,costs)],'supported':True}


def test_partial_coloring_enforces_nonmatches_and_does_not_create_ghost_nodes():
    e=[edge(0,1,[[(1,0)]],[0.])]
    a,info=solve_partial([2,2],3,e,[[0,1],[1,2]],seconds=5)
    assert actual_pairs(a,0,1)==((1,0),) and np.count_nonzero(a<0)==2
    assert info['ghost_nodes']==0 and info['real_nodes']==4


def test_partial_minimax_matches_exhaustive_small_graph():
    e=[edge(0,1,[[(0,0),(1,1)],[(0,1),(2,0)]],[.3,0]),
       edge(1,2,[[(0,0),(1,1)],[(0,1)],[(1,0)]],[0,.1,.2]),
       edge(2,0,[[(0,0),(1,1)],[(0,2),(1,0)]],[0,.4])]
    possibilities=[]
    for left,right in itertools.product(itertools.permutations(range(3),2),repeat=2):
        a=np.full((3,3),-1,int);a[0]=[0,1,2];a[1,list(left)]=[0,1];a[2,list(right)]=[0,1]
        m=partial_metrics(a,e)
        if not m['violated_factors']:possibilities.append((m['max_regret'],m['sum_model_cost']))
    a,info=solve_partial([3,2,2],3,e,[[0,1,2],[0,1],[0,1]],epsilon=1,seconds=5)
    assert a is not None and info['primary_optimal']
    m=partial_metrics(a,e)
    np.testing.assert_allclose((m['max_regret'],m['sum_model_cost']),min(possibilities),atol=1e-6)


def test_infeasible_relation_is_not_repaired_by_dropping_an_edge():
    e=[edge(0,1,[[(0,0),(1,1)]],[0]),edge(1,2,[[(0,0),(1,1)]],[0]),
       edge(2,0,[[(0,1),(1,0)]],[0])]
    a,info=solve_partial([2,2,2],2,e,[[0,1]]*3,seconds=5)
    assert a is None and info['status']=='mandatory_links_same_site_conflict' and len(e)==3


def test_invalid_partial_relation_and_nonfinite_input_are_rejected():
    with pytest.raises(ValueError):solve_partial([2,2],2,[edge(0,1,[[(0,0)]],[0])],[[0,1]]*2)
    with pytest.raises(ValueError):observed_error(np.array([np.nan]),np.array([1.]))


def test_fullsite_ablation_holds_root_equation_family_fixed():
    xy,zs=spectra();full=np.array([len(z)==3 for z in zs])
    a,info=fit_observed(xy,zs,3,mode='observed_fullsites')
    b,_=fit_observed(xy[full],[z for z in zs if len(z)==3],3)
    np.testing.assert_array_equal(a.mean,b.mean);np.testing.assert_array_equal(a.coefficients,b.coefficients)
    assert a.scale==b.scale and info['partial_training_sites_unused']==10


def test_zero_observations_do_not_make_vacancy_identity_nodes():
    e=[edge(0,1,[[]],[0.])]
    a,info=solve_partial([0,2],3,e,[[],[0,2]],seconds=5)
    assert info['real_nodes']==2 and info['ghost_nodes']==0
    assert np.all(a[0]<0) and np.count_nonzero(a[1]>=0)==2


