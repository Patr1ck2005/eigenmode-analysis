"""Global coloring of REAL observations with exact partial edge relations.

No ghost/vacancy node, synthetic frequency, discarded edge, or extra band is
introduced. Nonmatches are inequalities: an observed pair absent from the
selected relation cannot silently share a color across that edge.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .local_pair_stitching import Union


def actual_pairs(assignment, s, t):
    return tuple(sorted((int(i),int(j)) for i,j in zip(assignment[s],assignment[t]) if i>=0 and j>=0))


def partial_metrics(assignment, edges):
    violated=[];choices=[];costs=[];regrets=[];matches=[]
    for q,e in enumerate(edges):
        pairs=actual_pairs(assignment,*e['sites'])
        ids=[i for i,o in enumerate(e['options']) if pairs==tuple(sorted(map(tuple,o['pairs'])))]
        if not ids:violated.append(q);choices.append(None);continue
        chosen=ids[0];choices.append(chosen);matches.append(len(pairs))
        if e['supported']:
            cost=e['options'][chosen]['cost'];costs.append(cost)
            regrets.append(cost-min(o['cost'] for o in e['options']))
    return {'violated_factors':violated,'choices':choices,'max_regret':max(regrets,default=0.),
        'sum_model_cost':sum(costs),'max_model_cost':max(costs,default=0.),
        'matched_links_on_all_factors':sum(matches),
        'warning':'Cost only covers observed-to-observed paths; retain match-count and output-face changes separately.'}


def solve_partial(counts, k, edges, baseline_labels, *, epsilon=5e-5, seconds=20):
    counts=np.asarray(counts,int);n=len(counts)
    if n<1 or k<1 or np.any(counts<0) or np.any(counts>k) or epsilon<=0:
        raise ValueError('valid finite site multiplicities required')
    if len(baseline_labels)!=n or any(len(row)!=m or len(set(row))!=m or any(b<0 or b>=k for b in row)
                                    for row,m in zip(baseline_labels,counts)):
        raise ValueError('baseline must give one distinct old color to each observed node')
    offsets=np.r_[0,np.cumsum(counts)];nodes=int(offsets[-1]);union=Union(nodes)
    normalized=[]
    for e in edges:
        s,t=e['sites']
        if min(s,t)<0 or max(s,t)>=n or s==t:raise ValueError('invalid edge sites')
        options=[]
        for o in e['options']:
            pairs=tuple(sorted(map(tuple,o['pairs'])))
            if (len(set(pairs))!=len(pairs) or len({p[0] for p in pairs})!=len(pairs)
                or len({p[1] for p in pairs})!=len(pairs)
                or any(min(i,j)<0 or i>=counts[s] or j>=counts[t] for i,j in pairs)
                or not np.isfinite(o['cost']) or o['cost']<0):
                raise ValueError('valid partial injection and finite cost required')
            if counts[s]+counts[t]-len(pairs)>k:
                raise ValueError('partial relation requires more colors than the fixed block multiplicity')
            options.append({**o,'pairs':pairs})
        normalized.append({**e,'options':options})
        if len(options)==1:
            for i,j in options[0]['pairs']:union.join(int(offsets[s]+i),int(offsets[t]+j))
    edges=normalized
    if any(not e['options'] for e in edges):
        return None,{'status':'empty_partial_factor','factors':[i for i,e in enumerate(edges) if not e['options']]}
    roots=sorted({union.root(v) for v in range(nodes)});lookup={r:i for i,r in enumerate(roots)}
    comp=np.array([lookup[union.root(v)] for v in range(nodes)],int)
    at=[comp[offsets[s]:offsets[s+1]] for s in range(n)]
    for s,row in enumerate(at):
        if len(set(row))!=len(row):return None,{'status':'mandatory_links_same_site_conflict','site':s,'components':len(roots)}
    m=len(roots);nv=m*k;cursor=nv;choices={}
    for q,e in enumerate(edges):
        if len(e['options'])>1:choices[q]=cursor;cursor+=len(e['options'])
    maxvar=cursor;size=cursor+1
    ri=[];ci=[];data=[];lower=[];upper=[]
    def con(ids,weights,lo,hi):
        r=len(lower);ri.extend([r]*len(ids));ci.extend(ids);data.extend(weights);lower.append(lo);upper.append(hi)
    for q in range(m):con(list(range(q*k,(q+1)*k)),[1.]*k,1,1)
    for row in at:
        for b in range(k):con([int(q*k+b) for q in row],[1.]*len(row),0,1)
    objective2=np.zeros(size);lb=np.zeros(size);ub=np.ones(size);ub[maxvar]=np.inf
    for s,row in enumerate(at):
        for i,q in enumerate(row):
            for b in range(k):
                if b!=baseline_labels[s][i]:objective2[q*k+b]+=1e-8/max(nodes,1)
    for q,e in enumerate(edges):
        s,t=e['sites'];off=choices.get(q)
        if off is not None:con(list(range(off,off+len(e['options']))),[1.]*len(e['options']),1,1)
        for j,o in enumerate(e['options']):
            y=None if off is None else off+j;pairs=set(o['pairs'])
            for i,u in enumerate(at[s]):
                for h,v in enumerate(at[t]):
                    eq=(i,h) in pairs
                    if u==v:
                        if not eq:
                            if y is None:return None,{'status':'mandatory_nonmatch_conflict','factor':q,'pair':[i,h]}
                            ub[y]=0
                        continue
                    for b in range(k):
                        if eq:
                            if y is None:continue # Already contracted above.
                            con([u*k+b,v*k+b,y],[1.,-1.,1.],-np.inf,1)
                            con([u*k+b,v*k+b,y],[-1.,1.,1.],-np.inf,1)
                        else:
                            con([u*k+b,v*k+b]+([] if y is None else [y]),
                                [1.,1.]+([] if y is None else [1.]),-np.inf,1 if y is None else 2)
            if y is not None and e['supported']:objective2[y]=o['cost']/epsilon
        if off is not None:
            best=min(o['cost'] for o in e['options'])
            regret=[(o['cost']-best)/epsilon for o in e['options']]
            con(list(range(off,off+len(regret)))+[maxvar],regret+[-1.],-np.inf,0)
    anchor=int(np.argmax(counts))
    for i,q in enumerate(at[anchor]):lb[q*k+baseline_labels[anchor][i]]=1
    matrix=coo_matrix((data,(ri,ci)),shape=(len(lower),size)).tocsc()
    constraints=LinearConstraint(matrix,np.asarray(lower),np.asarray(upper))
    integral=np.ones(size);integral[maxvar]=0
    objective=np.zeros(size);objective[maxvar]=1
    solved=milp(objective,integrality=integral,bounds=Bounds(lb,ub),constraints=constraints,
                options={'time_limit':seconds,'mip_rel_gap':0})
    info={'real_nodes':nodes,'ghost_nodes':0,'components':m,'variables':size,'constraints':len(lower),
          'anchor':anchor,'primary_solver_status':int(solved.status),'primary_optimal':solved.status==0,
          'solver_message':solved.message,'stage_qualified':False,'supported_factors':sum(e['supported'] for e in edges)}
    if solved.x is None:return None,{**info,'status':'no_feasible_partial_numbering_returned'}
    chosen=solved
    if solved.status==0:
        ub[maxvar]=max(0.,solved.x[maxvar])+1e-7
        secondary=milp(objective2,integrality=integral,bounds=Bounds(lb,ub),constraints=constraints,
                       options={'time_limit':seconds,'mip_rel_gap':0})
        info.update(secondary_solver_status=int(secondary.status),secondary_optimal=secondary.status==0)
        if secondary.x is not None:chosen=secondary
    colors=np.argmax(chosen.x[:nv].reshape(m,k),axis=1)
    if np.min(np.max(chosen.x[:nv].reshape(m,k),axis=1),initial=1)<1-1e-6:raise RuntimeError('nonintegral colors')
    a=np.full((n,k),-1,int);baseline=np.full((n,k),-1,int)
    for s,row in enumerate(at):
        if len(set(colors[row]))!=len(row):raise RuntimeError('same-site candidate collision')
        for i,q in enumerate(row):a[s,colors[q]]=i;baseline[s,baseline_labels[s][i]]=i
    observed=partial_metrics(a,edges);old=partial_metrics(baseline,edges)
    if observed['violated_factors']:raise RuntimeError('partial equalities/nonmatches escaped validation')
    if not old['violated_factors'] and observed['max_regret']>old['max_regret']+1e-10:
        return baseline,{**info,'status':'worse_incumbent_rejected','baseline_metrics':old,'rejected_metrics':observed}
    return a,{**info,'status':'conditional_partial_grouping','metrics':observed,'baseline_metrics':old,
        'missing_slots':int(np.count_nonzero(a<0)),'changed_slot_entries':int(np.count_nonzero(a!=baseline)),
        'output_points_added':0,'parameter_factors_dropped':0}
