"""Scalar set models with real-valued branches by construction.

Unordered pairs/triples supply symmetric quantities, never complex-frequency
identities. Polynomial magnitude fitting recovers a signed smooth field without
reading the measured imaginary part. Held-out grid sites reject unsupported
models; this is a local research model family, not universal qualification.
"""
import numpy as np
from itertools import product
from scipy.linalg import qr
from scipy.optimize import linear_sum_assignment


def powers(degree):
    return [(i,d-i) for d in range(degree+1) for i in range(d+1)]


def basis(xy,degree):
    x,y=np.asarray(xy).T
    return np.column_stack([x**i*y**j for i,j in powers(degree)])


def signed_polynomial(xy,magnitude,degree):
    a=basis(xy,degree); n=a.shape[1]
    if len(a)<n or np.linalg.matrix_rank(a)<n:raise ValueError('insufficient independent polynomial samples')
    _,_,rows=qr(a.T,pivoting=True,mode='economic'); chosen=rows[:n]
    signs=1-2*((np.arange(1<<(n-1))[:,None]>>np.arange(n-1))&1)
    signs=np.column_stack([np.ones(len(signs)),signs])
    coefs=np.linalg.solve(a[chosen],(signs*magnitude[chosen]).T).T
    residual=abs(a@coefs.T)-magnitude[:,None]
    best=coefs[int(np.argmin(np.sum(residual*residual,axis=0)))]
    for _ in range(3):
        sign=np.where(a@best>=0,1.,-1.)
        best=np.linalg.lstsq(a,magnitude*sign,rcond=None)[0]
    return best,{'sign_hypotheses':len(signs),'magnitude_residual':float(np.max(abs(abs(a@best)-magnitude)))}


def _fit(xy,z,kind):
    mean=z.mean(axis=1);a2=basis(xy,2)
    result={'kind':kind,'mean':np.linalg.lstsq(a2,mean,rcond=None)[0].tolist()}
    if kind=='single':return result
    if kind=='pair-polynomial':
        coef,info=signed_polynomial(xy,.5*(z[:,1]-z[:,0]),2)
        return dict(result,seam=coef.tolist(),seam_degree=2,**info)
    if kind=='pair-root':
        d=(.5*(z[:,1]-z[:,0]))**2;a4=basis(xy,4)
        design=np.column_stack([a2*d[:,None],a4])
        coeff,_,rank,_=np.linalg.lstsq(design,d*d,rcond=None)
        if rank<design.shape[1]:raise ValueError('pair-root latent fields are not identifiable on these samples')
        w=a4@coeff[6:]
        if w.min() < -1e-8*max(float(np.max(d*d)),1e-8):
            raise ValueError('fitted squared field is negative at observations')
        seam,info=signed_polynomial(xy,np.sqrt(np.maximum(w,0.)),2)
        return dict(result,u=coeff[:6].tolist(),u_degree=2,seam=seam.tolist(),seam_degree=2,**info)
    if kind=='triple-root':
        centered=z-mean[:,None]
        u=4*np.prod(centered,axis=1)
        magnitude=4/(3*np.sqrt(3))*abs((z[:,0]-z[:,1])*(z[:,0]-z[:,2])*(z[:,1]-z[:,2]))
        seam,info=signed_polynomial(xy,magnitude,3)
        return dict(result,u=np.linalg.lstsq(basis(xy,3),u,rcond=None)[0].tolist(),
                    u_degree=3,seam=seam.tolist(),seam_degree=3,**info)
    raise ValueError('unknown scalar set model')


def _predict(model,xy,seam_mask=None,critical_mask=None):
    mean=basis(xy,2)@model['mean'];kind=model['kind']
    if kind=='single':return mean[:,None]
    v=basis(xy,model['seam_degree'])@model['seam']
    if seam_mask is not None:v=np.where(seam_mask,0.,v)
    if kind=='pair-polynomial':gap=abs(v)
    elif kind=='pair-root':
        u=basis(xy,model['u_degree'])@model['u'];norm=np.hypot(u,2*v)
        if critical_mask is not None:
            u=np.where(critical_mask,0.,u);v=np.where(critical_mask,0.,v);norm=np.hypot(u,2*v)
        d=np.where(u>=0,.5*(norm+u),np.divide(2*v*v,norm-u,out=np.zeros_like(u),where=norm-u>0))
        gap=np.sqrt(np.maximum(d,0.))
    elif kind=='triple-root':
        u=basis(xy,model['u_degree'])@model['u']
        if critical_mask is not None:u=np.where(critical_mask,0.,u);v=np.where(critical_mask,0.,v)
        radius=np.hypot(u,v)**(1/3);angle=np.arctan2(v,u)/3
        result=np.sort(mean[:,None]+radius[:,None]*np.cos(angle[:,None]+2*np.pi*np.arange(3)/3),axis=1)
        zero=v==0
        if np.any(zero):
            r=abs(u[zero])**(1/3)
            result[zero]=mean[zero,None]+np.where((u[zero]>=0)[:,None],r[:,None]*np.array([-.5,-.5,1.]),r[:,None]*np.array([-1.,.5,.5]))
        return result
    return np.column_stack([mean-gap,mean+gap])


def predict(model,coordinates,seam_mask=None,critical_mask=None):
    xy=(np.asarray(coordinates)-model['xy_center'])/model['xy_radius']
    return _predict(model,xy,seam_mask,critical_mask)*model['height_scale']+model['height_center']


def fit_checked(coordinates,heights,parity,kind,tolerance=1e-6):
    center=np.mean([coordinates.min(axis=0),coordinates.max(axis=0)],axis=0)
    radius=np.maximum(np.ptp(coordinates,axis=0)/2,1e-12)
    xy=(coordinates-center)/radius
    z0=float(np.mean(heights));zs=max(float(np.ptp(heights)),1e-10)
    z=(heights-z0)/zs
    folds=[]
    for fold in (0,1):
        training=parity!=fold;testing=~training
        model=_fit(xy[training],z[training],kind)
        error=float(np.max(abs(_predict(model,xy[testing])-z[testing])))
        folds.append(error)
    model=_fit(xy,z,kind)
    native=float(np.max(abs(_predict(model,xy)-z)))
    return dict(model,xy_center=center.tolist(),xy_radius=radius.tolist(),
        height_center=z0,height_scale=zs,native_relative_error=native,
        held_out_relative_errors=folds,accepted=max(native,*folds)<=tolerance,
        model_tolerance=tolerance,stage_qualified=False)


def _fit_rank_collection(c,f,part='real'):
    if c.grid_shape is None or c.coordinate_dimension!=2:raise ValueError('scalar models require a 2D grid')
    sizes={len(z) for z in f.values}
    m=max(sizes);orders=[np.argsort(getattr(z,part),kind='stable') for z in f.values]
    complete=np.array([len(z)==m for z in f.values])
    if m==0:return [],{'reason':'empty scalar data','accepted_groups':0}
    heights=np.asarray([getattr(z,part)[o] for z,o,keep in zip(f.values,orders,complete) if keep])
    parity=np.indices(c.grid_shape).sum(axis=0).ravel()%2
    proposals=[];attempts=[]
    for size,kinds in [(1,['single']),(2,['pair-polynomial','pair-root']),(3,['triple-root'])]:
        for start in range(m-size+1):
            for kind in kinds:
                try:
                    model=fit_checked(c.coordinates[complete],heights[:,start:start+size],parity[complete],kind)
                    model['ranks']=list(range(start,start+size))
                    attempts.append({k:v for k,v in model.items() if k not in ('mean','seam','u')})
                    if model['accepted']:proposals.append(model)
                except (ValueError,np.linalg.LinAlgError) as error:
                    attempts.append({'ranks':list(range(start,start+size)),'kind':kind,'accepted':False,'reason':str(error)})
    # Select non-overlapping groups, maximizing modeled ranks then preferring
    # the smaller model family. Uncovered raw ranks stay in the fallback mesh.
    complexity={'single':6,'pair-polynomial':12,'pair-root':18,'triple-root':26}
    dp={m:((0,0),[])}
    for start in range(m-1,-1,-1):
        choices=[dp[start+1]]
        for p in proposals:
            if p['ranks'][0]!=start:continue
            score,following=dp[p['ranks'][-1]+1]
            choices.append(((score[0]+len(p['ranks']),score[1]-complexity[p['kind']]),[p]+following))
        dp[start]=max(choices,key=lambda row:row[0])
    models=dp[0][1]
    modeled={rank for model in models for rank in model['ranks']}
    remaining=[rank for rank in range(m) if rank not in modeled]
    if modeled and remaining and len(sizes)==1:
        from .model import CandidateFibers
        reduced=CandidateFibers(tuple(z[o[remaining]] for z,o in zip(f.values,orders)))
        extra,child=_fit_rank_collection(c,reduced,part)
        for model in extra:model['ranks']=[remaining[rank] for rank in model['ranks']]
        for attempt in child.get('attempts',[]):
            if 'ranks' in attempt:attempt['ranks']=[remaining[rank] for rank in attempt['ranks']]
            attempt['after_removing_validated_groups']=True
        models+=extra;attempts+=child.get('attempts',[])
    # At incomplete sites, identify only model values actually supported by
    # an original scalar. The assignment is injective, and no missing rank
    # becomes a measurement or a supported rendering corner.
    columns=[(mid,band) for mid,model in enumerate(models) for band in range(len(model['ranks']))]
    predicted=np.column_stack([predict(model,c.coordinates) for model in models]) if models else np.empty((c.n_sites,0))
    limits=np.array([model['model_tolerance']*model['height_scale'] for model in models for _ in model['ranks']])
    references=[np.full((c.n_sites,len(model['ranks'])),-1,dtype=int) for model in models]
    for site,z in enumerate(f.values):
        if complete[site]:
            for mid,model in enumerate(models):references[mid][site]=orders[site][model['ranks']]
        elif len(z) and columns:
            cost=abs(predicted[site,:,None]-getattr(z,part)[None,:])
            # Reject unavailable assignments before optimization so a bad
            # prediction cannot steal a supported observation from another.
            valid=cost<=np.maximum(limits[:,None],1e-11)
            augmented=np.column_stack([np.where(valid,cost/np.maximum(limits[:,None],1e-11),1e6),
                np.full((len(columns),len(columns)),2.)])
            a,b=linear_sum_assignment(augmented)
            for i,j in zip(a,b):
                if j<len(z) and valid[i,j]:
                    mid,band=columns[i];references[mid][site,band]=j
    for model,refs in zip(models,references):model['references']=refs.tolist()
    return models,{'attempts':attempts,'accepted_groups':len(models),
        'modeled_ranks':sum(len(p['ranks']) for p in models),'total_ranks':m,
        'complete_fit_sites':int(complete.sum()),'incomplete_sites':int((~complete).sum()),
        'other_frequency_projection_used':False,'physical_fields_used':False,'stage_qualified':False}


def _polynomial_peel(c,f,part,seed_mesh=None):
    """Extract supported polynomial graphs across changing height ranks.

    Coordinate anchors or local stencils propose graphs. Every site tests
    them; support must span both axes. Greedy maximum support removes broad
    smooth graphs before considering partial graphs, with raw multiplicity
    retained by repeated injective extraction. No identity labels are read.
    """
    remaining=[list(range(len(z))) for z in f.values]
    available=np.array([len(z)>0 for z in f.values]);sites=np.flatnonzero(available)
    if len(sites)<12:return [],remaining,{'reason':'insufficient 2D support'}
    center=np.mean([c.coordinates.min(axis=0),c.coordinates.max(axis=0)],axis=0)
    radius=np.maximum(np.ptp(c.coordinates,axis=0)/2,1e-12)
    xy=(c.coordinates-center)/radius;degree=1 if seed_mesh is None else 2;a=basis(xy,degree)
    if np.linalg.matrix_rank(a[sites])<3:return [],remaining,{'reason':'affine coordinate rank'}
    if seed_mesh is None:
        _,_,pivot=qr(a[sites].T,pivoting=True,mode='economic');anchors=sites[pivot[:3]]
        hypothesis_count=int(np.prod([len(f.values[s]) for s in anchors]))
        if hypothesis_count>16384:return [],remaining,{'reason':'bounded affine hypothesis budget','hypotheses':hypothesis_count}
        samples=np.array(list(product(*[np.sort(getattr(f.values[s],part)) for s in anchors])))
        coefficients=np.linalg.solve(a[anchors],samples.T).T
    else:
        # A cross of five connected observations plus a diagonal supplies a
        # quadratic proposal. Both routes to the diagonal compete; global raw
        # support, not the seed's labels, decides whether either is usable.
        links={}
        for edge in seed_mesh.edges:
            links[edge.source.site,edge.target.site,edge.source.local]=edge.target.local
            links[edge.target.site,edge.source.site,edge.target.local]=edge.source.local
        nx,ny=c.grid_shape;proposals=[]
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                s=i*ny+j;around=[s,s-ny,s+ny,s-1,s+1];diagonal=s+ny+1
                for local in range(len(f.values[s])):
                    ls=[local]+[links.get((s,target,local)) for target in around[1:]]
                    if any(value is None for value in ls):continue
                    choices={links.get((s+ny,diagonal,ls[2])),links.get((s+1,diagonal,ls[4]))}-{None}
                    for other in choices:
                        ss=around+[diagonal];jj=ls+[other]
                        proposals.append(np.linalg.solve(a[ss],np.array([getattr(f.values[q],part)[v] for q,v in zip(ss,jj)])))
        if not proposals:return [],remaining,{'reason':'no complete quadratic support stencil'}
        coefficients=np.asarray(proposals);hypothesis_count=len(proposals)
    values=np.concatenate([getattr(z,part) for z in f.values]);height_center=float(np.mean(values))
    height_scale=max(float(np.ptp(values)),1e-10)
    coefficients[:,0]-=height_center;coefficients/=height_scale
    coefficients=np.unique(np.round(coefficients,13),axis=0)
    observed=np.full((c.n_sites,max(map(len,f.values))),np.inf)
    for site,z in enumerate(f.values):observed[site,:len(z)]=(getattr(z,part)-height_center)/height_scale
    minimum_support=max(12,int(np.ceil(.15*available.sum())))
    tolerance=1e-9;ranked=[]
    counts=np.fromiter(map(len,f.values),dtype=int)
    complete=counts==counts.max()
    for first in range(0,len(coefficients),64):
        coefs=coefficients[first:first+64];predicted=a@coefs.T
        residual=np.min(abs(predicted[:,:,None]-observed[:,None,:]),axis=2)
        for local in range(len(coefs)):
            support=np.flatnonzero(residual[:,local]<=tolerance)
            if len(support)<minimum_support:continue
            if np.any(residual[complete,local]>tolerance):continue
            if any(len(np.unique(c.coordinates[support,axis]))<3 for axis in range(2)):continue
            ranked.append(coefs[local])
    models=[];parity=np.indices(c.grid_shape).sum(axis=0).ravel()%2
    while ranked:
        counts=np.isfinite(observed).sum(axis=1)
        if not counts.max():break
        complete=counts==counts.max()
        choices=[]
        for coef in ranked:
            prediction=a@coef;difference=abs(observed-prediction[:,None])
            local=np.argmin(difference,axis=1);error=difference[np.arange(c.n_sites),local]
            supported=np.flatnonzero(error<=tolerance)
            if len(supported)<minimum_support:continue
            # Incidental planes through a few rows of a curved band cannot
            # cut it into artificial finite pieces. Full-count sites must
            # all support a graph before it is removed from the remainder.
            if np.any(error[complete]>tolerance):continue
            if any(len(np.unique(c.coordinates[supported,axis]))<3 for axis in range(2)):continue
            choices.append((len(supported),-float(error[supported].sum()),tuple(coef),local,supported))
        if not choices:break
        _,_,coef,local,supported=max(choices,key=lambda row:row[:3])
        raw_values=np.array([getattr(f.values[s],part)[local[s]] for s in supported])[:,None]
        try:model=fit_checked(c.coordinates[supported],raw_values,parity[supported],'single')
        except (ValueError,np.linalg.LinAlgError):
            ranked=[v for v in ranked if tuple(v)!=coef];continue
        if not model['accepted']:
            ranked=[v for v in ranked if tuple(v)!=coef];continue
        references=np.full((c.n_sites,1),-1,dtype=int);references[supported,0]=local[supported]
        model.update(references=references.tolist(),ranks=[0],rank_space='polynomial graph across input ranks',
            proposal=('three-anchor affine' if degree==1 else 'cross-and-diagonal quadratic')+' verified against raw 2D support',
            support_count=len(supported))
        models.append(model)
        for site in supported:
            observed[site,local[site]]=np.inf;remaining[site].remove(int(local[site]))
    return models,remaining,{'degree':degree,'hypotheses':hypothesis_count,'supported_hypotheses':len(ranked),
        'selected_graphs':len(models),'minimum_support':minimum_support,'relative_match_tolerance':tolerance,
        'support_rule':'all sites with the current maximum observed candidate count'}


def fit_collection(c,f,part='real'):
    if c.grid_shape is None or c.coordinate_dimension!=2:raise ValueError('scalar models require a 2D grid')
    affine,remaining,peeling=_polynomial_peel(c,f,part)
    from .model import CandidateFibers
    reduced=CandidateFibers(tuple(z[ids] for z,ids in zip(f.values,remaining)))
    models,info=_fit_rank_collection(c,reduced,part)
    for model in models:
        model['references']=[[remaining[site][local] if local>=0 else -1 for local in row]
            for site,row in enumerate(model['references'])]
        model['rank_space']='remaining ranks after affine extraction'
    models=affine+models
    claimed=[{local for model in models for local in model['references'][site] if local>=0} for site in range(c.n_sites)]
    residual=[np.array([local for local in range(len(z)) if local not in claimed[site]],dtype=int) for site,z in enumerate(f.values)]
    if max(map(len,residual),default=0)>1:
        from .scalar_strip import reconstruct_scalar,StripConfig
        leftover=CandidateFibers(tuple(z[ids] for z,ids in zip(f.values,residual)))
        seeds,_=reconstruct_scalar(c,leftover,part=part,config=StripConfig(objective='angle'))
        extra,_,report=_polynomial_peel(c,leftover,part,seed_mesh=seeds)
        for model in extra:
            model['references']=[[int(residual[site][local]) if local>=0 else -1 for local in row]
                for site,row in enumerate(model['references'])]
        models+=extra;info['quadratic_extraction']=report
    info.update(affine_extraction=peeling,accepted_groups=len(models),total_raw_candidates=f.n_candidates,
        matched_raw_candidates=sum(sum(local>=0 for local in row) for model in models for row in model['references']))
    return models,info
