"""Numerical subset of frozen Stable v1; see migration provenance."""
from collections import defaultdict
from .local_pair_stitching import Union


def interaction_blocks(baseline,patches,pref):
    labels={(s,int(j)):b for s,row in enumerate(baseline) for b,j in enumerate(row) if j>=0}
    u=Union(baseline.shape[1])
    for a,b,_ in pref:u.join(labels[a],labels[b])
    for p in patches:
        ids=sorted({labels[tuple(r)] for branch in p['branches'] for r in branch})
        for i in ids[1:]:u.join(ids[0],i)
    blocks=defaultdict(list)
    for b in range(baseline.shape[1]):blocks[u.root(b)].append(b)
    return sorted(blocks.values(),key=lambda b:b[0])
