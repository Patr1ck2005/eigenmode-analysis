"""Complete raw-band grouping from synchronized neighbor permutations.

Two fixed connection hypotheses (proximity / secant) have identical Re-only
and complex-frequency arms. Unlike a partial edge cover, each site assigns
all candidates to maximum-multiplicity band slots before ordinary plotting.
No truth, payload, original indices, added observations or spectral models.
"""
import time
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment

from .classification import validate_assignment, mesh_from_assignment


def edge_costs(c, values, kind):
    if kind not in ('proximity','secant'):
        raise ValueError('unknown transport hypothesis')
    m=max(map(len,values),default=0)
    costs=[]; provenance=[]
    lookup=c.edge_index()
    # Both channels share the scale computed from Re only. Im stays in the
    # same physical frequency units; no whitening of the weak component.
    flat=np.concatenate(values); scale=max(float(np.ptp(flat.real)),1e-12)
    normalized=tuple(z/scale for z in values)
    for a,b in c.edges:
        za,zb=normalized[a],normalized[b]
        raw=abs(za[:,None]-zb[None,:])**2
        terms=[]
        if kind=='secant' and c.grid_shape is not None:
            ia=np.asarray(c.grid_index(a));ib=np.asarray(c.grid_index(b))
            axis=np.flatnonzero(ib!=ia)
            if len(axis)==1:
                axis=int(axis[0]);step=ib-ia
                for middle,other,index,direction,transpose in ((a,b,ia,-step,False),(b,a,ib,step,True)):
                    ij=index+direction
                    if np.any(ij<0) or np.any(ij>=c.grid_shape):continue
                    previous=c.site_index(ij)
                    if tuple(sorted((previous,middle))) not in lookup or not len(normalized[previous]):continue
                    h=c.coordinates[other,axis]-c.coordinates[middle,axis]
                    hp=c.coordinates[middle,axis]-c.coordinates[previous,axis]
                    prediction=normalized[middle][:,None]+h/hp*(normalized[middle][:,None]-normalized[previous][None,:])
                    error=abs(prediction[:,:,None]-normalized[other][None,None,:])**2
                    score=error.min(axis=1)
                    terms.append(score.T if transpose else score)
        if terms: raw=sum(terms)/len(terms)
        matrix=np.zeros((m,m))
        matrix[:len(za),:len(zb)]=raw
        # A fixed-cardinality edge match. Crossing an absent slot costs more
        # than all real pair costs combined; dummy/dummy matching costs zero.
        penalty=(m+1)*max(float(raw.max(initial=0.)),1.)
        matrix[:len(za),len(zb):]=penalty
        matrix[len(za):,:len(zb)]=penalty
        costs.append(matrix)
        provenance.append({'secant_stencils':len(terms)})
    return costs,provenance


def classify(c, values, *, channel='real', kind='proximity', max_passes=20, cost_transform=None):
    tick=time.perf_counter()
    if channel not in ('real','complex') or len(values)!=c.n_sites:
        raise ValueError('invalid channel or site count')
    arrays=tuple(np.asarray(z).copy() for z in values)
    if any(z.ndim!=1 or not np.isfinite(z).all() for z in arrays):
        raise ValueError('finite original candidate arrays required')
    if channel=='real':arrays=tuple(z.real.astype(complex) for z in arrays)
    orders=tuple(np.lexsort((z.imag,z.real)) for z in arrays)
    arrays=tuple(z[o] for z,o in zip(arrays,orders));m=max(map(len,arrays),default=0)
    if not m:return np.empty((c.n_sites,0),int),{'status':'empty_input'}
    costs,details=edge_costs(c,arrays,kind)
    transform_info=None
    if cost_transform is not None:
        costs,transform_info=cost_transform(c,arrays,orders,costs)
    adjacency=[[] for _ in arrays];connections=[]
    for eid,((a,b),cost) in enumerate(zip(c.edges,costs)):
        rows,cols=linear_sum_assignment(cost);p=np.empty(m,int);p[rows]=cols
        best=float(cost[rows,cols].sum());second=np.inf
        for i,j in zip(rows,cols):
            # Dummy identities are not observations; do not let arbitrary
            # swaps of empty slots determine the edge's reliability.
            if i>=len(arrays[a]) or j>=len(arrays[b]):continue
            changed=cost.copy();changed[i,j]=np.inf
            try:
                r2,c2=linear_sum_assignment(changed)
                second=min(second,float(changed[r2,c2].sum()))
            except ValueError:pass
        margin=second-best if np.isfinite(second) else 0.
        confidence=margin/max(float(np.ptp(cost[:len(arrays[a]),:len(arrays[b])])) if len(arrays[a]) and len(arrays[b]) else 0.,1e-12)
        connections.append((confidence,eid,p))
        adjacency[a].append((b,eid,False));adjacency[b].append((a,eid,True))
    parent=list(range(c.n_sites));tree=[[] for _ in arrays]
    def root(s):
        while parent[s]!=s:
            parent[s]=parent[parent[s]];s=parent[s]
        return s
    for confidence,eid,p in sorted(connections,key=lambda r:(-r[0],r[1])):
        a,b=c.edges[eid];ra,rb=root(a),root(b)
        if ra==rb:continue
        parent[rb]=ra;tree[a].append((b,p));tree[b].append((a,np.argsort(p)))
    slots=np.full((c.n_sites,m),-1,int);anchors=[]
    for seed in sorted(range(c.n_sites),key=lambda s:(-len(arrays[s]),s)):
        if slots[seed,0]>=0:continue
        anchors.append(seed);slots[seed]=np.arange(m);queue=deque([seed])
        while queue:
            a=queue.popleft()
            for b,p in tree[a]:
                if slots[b,0]>=0:continue
                slots[b]=p[slots[a]];queue.append(b)
    def energy():
        return float(sum(costs[e][slots[a],slots[b]].sum() for e,(a,b) in enumerate(c.edges)))
    initial=energy();trace=[initial];converged=False
    # Gauss-Seidel permutation synchronization, no missing rows or cuts.
    for iteration in range(max_passes):
        changes=0
        for s in range(c.n_sites):
            if s in anchors:continue
            total=np.zeros((m,m)) # bands x local slots
            for other,eid,reverse in adjacency[s]:
                cost=costs[eid].T if reverse else costs[eid]
                total+=cost[:,slots[other]].T
            bands,locals_=linear_sum_assignment(total)
            before=float(total[np.arange(m),slots[s]].sum());after=float(total[bands,locals_].sum())
            if after < before-1e-12*max(1.,abs(before)):
                slots[s,bands]=locals_;changes+=1
        trace.append(energy())
        if not changes:converged=True;break
    assignment=np.full_like(slots,-1)
    for s,row in enumerate(slots):
        for band,local in enumerate(row):
            if local<len(arrays[s]):assignment[s,band]=orders[s][local]
    validate_assignment(c,values,assignment)
    if trace[-1]>trace[0]+1e-8*max(1.,abs(trace[0])):raise RuntimeError('synchronization energy increased')
    disagreements=0
    for _,eid,p in connections:
        a,b=c.edges[eid]
        for i,j in zip(slots[a],slots[b]):
            if i<len(arrays[a]) and j<len(arrays[b]) and p[i]!=j:disagreements+=1
    return assignment,{'version':'permutation-transport-v1','channel':channel,'kind':kind,
        'status':'fixed_point' if converged else 'pass_budget','converged':converged,'classification_succeeded':True,
        'max_passes':max_passes,'passes':iteration+1,'energy_trace':trace,'groups':m,
        'raw_candidates':sum(map(len,arrays)),'anchors':anchors,'edge_preference_disagreements':disagreements,
        'secant_supported_edges':sum(d['secant_stencils']>0 for d in details),'seconds':time.perf_counter()-tick,
        'connection_transform':transform_info,
        'input_policy':'coordinates, topology and Re'+(' plus measured Im' if channel=='complex' else ' only'),
        'physical_payload_used':False,'source_indices_used':False,'generated_measurements':0,
        'stage_qualified':False,'production_default':False,
        'limitations':['Local fixed point, not a global optimum or calibrated confidence.',
          'Maximum multiplicity and maximum adjacent cardinality are hypotheses.',
          'Complex monodromy may frustrate global labels; no branch cut is hidden or removed.',
          'Secant consistency can prefer wrong avoided-crossing continuations.']}


def reconstruct(c,f,**options):
    assignment,info=classify(c,f.values,**options)
    return mesh_from_assignment(c,f,assignment,info),info
