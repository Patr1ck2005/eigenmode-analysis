"""Local unordered-pair normal forms choose complete-band connections.

Reuses the known discriminant/root-continuation idea, now on 2D local patches
and with measured complex frequencies when explicitly selected. This is a
conditional model, not identification of the physical Hamiltonian or truth.
"""
from itertools import product
import numpy as np
from scipy.linalg import qr
from scipy.optimize import linear_sum_assignment

from . import permutation_transport as transport
from .classification import mesh_from_assignment


def design(xy,degree=2):
    x,y=np.asarray(xy).T
    return np.column_stack([np.ones(len(x)),x,y] + ([x*x,x*y,y*y] if degree==2 else []))


def signed_fit(xy,d):
    """Fit an unordered ±complex gap to one signed polynomial, no labels."""
    for degree in (1,2):
        a=design(xy,degree);p=a.shape[1]
        if len(a)<2*p+1 or np.linalg.matrix_rank(a)!=p:continue
        _,_,pivots=qr(a.T,pivoting=True,mode='economic');chosen=pivots[:p]
        signs=np.array([(1.,)+bits for bits in product((-1.,1.),repeat=p-1)])
        coefficients=np.linalg.solve(a[chosen],(signs*d[chosen]).T).T
        predictions=a@coefficients.T
        losses=np.minimum(abs(predictions-d[:,None]),abs(predictions+d[:,None]))
        selected=np.argsort(np.max(losses,axis=0))[:4]
        for k in selected:
            coefficient=coefficients[k]
            for _ in range(4):
                pred=a@coefficient
                signed=np.where(abs(pred-d)<=abs(pred+d),d,-d)
                coefficient=np.linalg.lstsq(a,signed,rcond=None)[0]
            residual=np.minimum(abs(a@coefficient-d),abs(a@coefficient+d))
            # Roundoff-level polynomial support only. This is not a noise
            # threshold or a claim that approximate fits prove zero gap.
            tolerance=256*np.finfo(float).eps*max(1.,np.linalg.cond(a))*max(float(abs(d).max()),1e-15)
            if float(residual.max())<=tolerance:
                return {'degree':degree,'coefficient':coefficient,'residual':float(residual.max()),'tolerance':tolerance}
    return None


def path_minimum(poly):
    norm=np.polynomial.polynomial.polymul(poly,np.conjugate(poly)).real
    stationary=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(norm))
    candidates=[0.,1.]+[float(r.real) for r in stationary if abs(r.imag)<1e-8 and 0<r.real<1]
    return min(abs(np.polynomial.polynomial.polyval(t,poly)) for t in candidates)


def choose_connection(xy,d,ends):
    """Return whether observed d at the edge ends have the same lifted sign."""
    center=(np.min(xy,axis=0)+np.max(xy,axis=0))/2
    radius=np.maximum(np.ptp(xy,axis=0)/2,1e-15)
    points=(xy-center)/radius;endxy=(ends[:,:2]-center)/radius
    da,db=ends[:,2]
    signed=signed_fit(points,d)
    if signed is not None:
        pred=design(endxy,signed['degree'])@signed['coefficient']
        if min(abs(pred))<=10*signed['tolerance']:
            return None,{'reason':'coincident endpoint'}
        sa=1 if abs(pred[0]-da)<=abs(pred[0]+da) else -1
        sb=1 if abs(pred[1]-db)<=abs(pred[1]+db) else -1
        return sa==sb,{'model':'signed-polynomial','degree':signed['degree'],'residual':signed['residual']}
    a=design(points);q=d*d
    if len(q)<13 or np.linalg.matrix_rank(a)!=6:return None,{'reason':'insufficient local 2D support'}
    coefficient=np.linalg.lstsq(a,q,rcond=None)[0];fitted=a@coefficient
    hat=np.sum(a*np.linalg.pinv(a).T,axis=1)
    loo=(fitted-q)/np.maximum(1-hat,1e-12)
    rounding=256*np.finfo(float).eps*max(1.,np.linalg.cond(a))*max(float(abs(q).max()),1e-30)
    error=max(float(abs(loo).max()),rounding)
    mid=(endxy[0]+endxy[1])/2
    q0,qm,q1=design(np.array([endxy[0],mid,endxy[1]]))@coefficient
    poly=np.array([q0,4*qm-3*q0-q1,2*(q0+q1-2*qm)])
    scale=float(abs(poly).max())
    while len(poly)>1 and abs(poly[-1])<64*np.finfo(float).eps*scale:poly=poly[:-1]
    minimum=float(path_minimum(poly))
    if minimum<=4*error:
        return None,{'reason':'path approaches model uncertainty disk','minimum_q':minimum,'loo_error':error}
    roots=np.polynomial.polynomial.polyroots(poly)
    delta=sum(float(np.angle((1-r)/(-r))) for r in roots)
    predicted=da*np.sqrt(abs(q1/q0))*np.exp(.5j*delta)
    same=abs(predicted-db)<=abs(predicted+db)
    return bool(same),{'model':'quadratic-discriminant','minimum_q':minimum,'loo_error':error,'argument_change':delta}


def classify(c,values,*,channel='complex',max_passes=20):
    if channel not in ('real','complex','guarded'):
        raise ValueError('unknown information channel')
    base_channel='real' if channel=='guarded' else channel
    seed,seed_info=transport.classify(c,values,channel=base_channel,kind='proximity',max_passes=max_passes)
    def modify(c,arrays,orders,costs):
        secant,_=transport.edge_costs(c,arrays,'secant')
        model_arrays=(tuple(np.asarray(z,complex)[order] for z,order in zip(values,orders))
                      if channel=='guarded' else arrays)
        alternatives=[secant]
        if channel=='guarded':alternatives.append(transport.edge_costs(c,model_arrays,'proximity')[0])
        canonical=np.full_like(seed,-1)
        for s,row in enumerate(seed):
            inverse=np.argsort(orders[s]);present=row>=0;canonical[s,present]=inverse[row[present]]
        attempts=[];changed=0;scale=max(float(np.ptp(np.concatenate(arrays).real)),1e-12)
        for eid,(a,b) in enumerate(c.edges):
            _,p=linear_sum_assignment(costs[eid]);inverse=np.argsort(p)
            pairs=[]
            for other_costs in alternatives:
                _,s=linear_sum_assignment(other_costs[eid]);relative=inverse[s]
                for i,j in enumerate(relative):
                    if i<int(j) and relative[j]==i:pairs.append((i,int(j)))
            used=set()
            for i,j in pairs:
                if i in used or j in used:continue
                used.update((i,j))
                if i>=len(arrays[a]) or j>=len(arrays[a]) or max(p[i],p[j])>=len(arrays[b]):continue
                bands=[int(np.flatnonzero(canonical[a]==local)[0]) for local in (i,j)]
                if set(canonical[b,bands])!={p[i],p[j]}:continue
                ia=np.asarray(c.grid_index(a));ib=np.asarray(c.grid_index(b))
                lo=np.maximum(np.minimum(ia,ib)-2,0);hi=np.minimum(np.maximum(ia,ib)+3,c.grid_shape)
                sites=[c.site_index((u,v)) for u in range(lo[0],hi[0]) for v in range(lo[1],hi[1])]
                sites=[t for t in sites if np.all(canonical[t,bands]>=0)]
                d=np.array([.5*(model_arrays[t][canonical[t,bands[0]]]-model_arrays[t][canonical[t,bands[1]]]) for t in sites])
                da=.5*(model_arrays[a][i]-model_arrays[a][j]);db=.5*(model_arrays[b][p[i]]-model_arrays[b][p[j]])
                ends=np.column_stack([c.coordinates[[a,b]],np.array([da,db])])
                same,info=choose_connection(c.coordinates[sites],d,ends)
                record={'edge':eid,'left_pair':[i,j],'seed_bands':bands,'same_as_proximity':same,**info}
                attempts.append(record)
                if same is None:continue
                distance=.5*(abs(2*da/scale)**2+abs(2*db/scale)**2)
                wanted=[p[i],p[j]] if same else [p[j],p[i]]
                for left,target in zip((i,j),wanted):
                    for right in (p[i],p[j]):costs[eid][left,right]=0. if right==target else distance
                changed+=int(not same)
        return costs,{'attempts':attempts,'changed_edge_pairs':changed,'seed':seed_info,
                      'rejected_pairs_keep_proximity':True,'local_model_role':'conditional edge permutation, no model values exported'}
    assignment,info=transport.classify(c,values,channel=base_channel,kind='proximity',max_passes=max_passes,cost_transform=modify)
    info.update(version='local-discriminant-transport-v1',family='2D local unordered normal-form transport',
                channel=channel,input_policy=('Re proximity plus measured-complex local-model evidence only' if channel=='guarded' else info['input_policy']),
                limitation='Empirical local model support, not proof between samples. Pair membership is seeded; multiway disagreements remain unsupported.')
    return assignment,info


def reconstruct(c,f,**options):
    assignment,info=classify(c,f.values,**options)
    return mesh_from_assignment(c,f,assignment,info),info
