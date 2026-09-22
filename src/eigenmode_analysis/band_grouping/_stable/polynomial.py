"""Numerical subset of frozen Stable v1; see migration provenance."""
import numpy as np


def design(xy,degree=2):
    x,y=np.asarray(xy).T
    return np.column_stack([x**i*y**(d-i) for d in range(degree+1) for i in range(d+1)])
