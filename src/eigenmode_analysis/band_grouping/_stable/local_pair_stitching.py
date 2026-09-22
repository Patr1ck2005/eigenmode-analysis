"""Numerical subset of frozen Stable v1; see migration provenance."""
from scipy.optimize import linear_sum_assignment


class Union:
    def __init__(self,n):self.parent=list(range(n))
    def root(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def join(self,a,b):self.parent[self.root(b)]=self.root(a)


def preferences(c,values):
    result=[]
    for eid,(a,b) in enumerate(c.edges):
        ia,ib=linear_sum_assignment(abs(values[a][:,None]-values[b][None,:])**2)
        result.extend(((a,int(i)),(b,int(j)),eid) for i,j in zip(ia,ib))
    return result
