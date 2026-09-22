"""Numerical regressions ported from the SHA-pinned Stable v1 release."""
import numpy as np
import pytest
from eigenmode_analysis.band_grouping._stable.local_spectral_contact import fit, predict, compare, fit_withheld, contacts


def mesh():
    return np.array([(x,y) for x in np.linspace(-1,1,6) for y in np.linspace(-1,1,5)])


def matrix_roots(xy):
    values=[]
    for x,y in xy:
        h=np.array([[.4+.08*x+.02j,.011,.006*y],
                    [.011,.48-.06*x+.012j,.014],
                    [.006*y,.014,.55+.03*y+.004j]])
        values.append(np.linalg.eigvals(h))
    return np.asarray(values)


def test_affine_three_mode_spectrum_is_reproduced():
    xy=mesh();z=matrix_roots(xy);model=fit(xy,z,3)
    query=np.array([[.17,.21],[-.34,.08]])
    assert max(compare(p,v)[1] for p,v in zip(predict(model,query),matrix_roots(query)))<1e-11


def test_permutations_and_common_frequency_shift_do_not_change_roots():
    xy=mesh();z=matrix_roots(xy);permuted=z.copy()
    for i in range(len(z)):permuted[i]=np.roll(permuted[i],i%3)
    shift=.17+.02*xy[:,0]-.01j*xy[:,1]
    a,b=fit(xy,z),fit(xy,permuted+shift[:,None])
    assert max(compare(p+v,q)[1] for p,v,q in zip(predict(a,xy),shift,predict(b,xy)))<1e-11


def test_withheld_values_do_not_enter_fit_or_normalization():
    xy=mesh();z=matrix_roots(xy);held=[12,17]
    a,_=fit_withheld(xy,z,held);changed=z.copy();changed[held]+=30+20j
    b,_=fit_withheld(xy,changed,held)
    np.testing.assert_array_equal(a.mean,b.mean)
    np.testing.assert_array_equal(a.coefficients,b.coefficients)
    assert a.scale==b.scale


def test_nonfinite_and_insufficient_sampling_are_rejected():
    xy=mesh();z=matrix_roots(xy);z[0,0]=np.nan
    with pytest.raises(ValueError):fit(xy,z)
    with pytest.raises(ValueError):fit(xy[:10],matrix_roots(xy[:10]))


def test_real_crossing_with_distinct_imaginary_parts_has_model_contact():
    xy=mesh();z=np.column_stack((1+.2*xy[:,0]+.03j,1-.2*xy[:,0]-.03j))
    model=fit(xy,z);ends=np.array([[-.4,0],[.4,0]])
    raw=np.column_stack((1+.2*ends[:,0]+.03j,1-.2*ends[:,0]-.03j))
    result=contacts(model,ends,raw)
    assert result['stable_numerical_continuation']
    assert len(result['contacts'])==1
    assert abs(result['contacts'][0]['t']-.5)<1e-9
    assert abs(result['contacts'][0]['imaginary_separation']-.06)<1e-9


def test_avoided_real_crossing_has_no_model_contact():
    xy=mesh();gap=np.sqrt(xy[:,0]**2+.04)
    z=np.column_stack((1+gap+.01j,1-gap+.01j));model=fit(xy,z)
    ends=np.array([[-.4,0],[.4,0]]);g=np.sqrt(ends[:,0]**2+.04)
    result=contacts(model,ends,np.column_stack((1+g+.01j,1-g+.01j)))
    assert result['stable_numerical_continuation']
    assert not result['contacts']


def test_complex_root_collision_is_not_claimed_as_resolved():
    xy=mesh();z=np.column_stack((1+.2*xy[:,0]+.01j,1-.2*xy[:,0]+.01j))
    model=fit(xy,z);ends=np.array([[-.4,0],[.4,0]])
    raw=np.column_stack((1+.2*ends[:,0]+.01j,1-.2*ends[:,0]+.01j))
    result=contacts(model,ends,raw)
    assert not result['stable_numerical_continuation']
    assert result['status']=='unresolved_numerical_root_continuation'

