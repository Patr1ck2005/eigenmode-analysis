"""Numerical subset of frozen Stable v1; see migration provenance."""
import numpy as np
from .model import AssociationEdge, BandCoverMesh, CandidateRef, CoverFace
from .tracking import _assemble_components


def validate_assignment(c, values, assignment):
    a = np.asarray(assignment)
    if a.ndim != 2 or a.shape[0] != c.n_sites or a.dtype.kind not in 'iu':
        raise ValueError('classification must be an integer site-by-band matrix')
    for site, row in enumerate(a):
        if np.any(row < -1) or sorted(row[row >= 0].tolist()) != list(range(len(values[site]))):
            raise ValueError('each original candidate must occur exactly once at its site')
    return a


def mesh_from_assignment(c, f, assignment, info=None):
    assignment = validate_assignment(c, f.values, assignment)
    edges, faces = [], []
    for eid, (a, b) in enumerate(c.edges):
        for i, j in zip(assignment[a], assignment[b]):
            if i >= 0 and j >= 0:
                edges.append(AssociationEdge(eid, CandidateRef(a, int(i)), CandidateRef(b, int(j)),
                                             0., .5, False, ()))
    for fid, sites in enumerate(c.faces):
        for row in assignment[list(sites)].T:
            if np.all(row >= 0):
                faces.append(CoverFace(fid, tuple(CandidateRef(s, int(i)) for s, i in zip(sites, row)), .5, False))
    refs = tuple(tuple(CandidateRef(s, j) for j in range(len(z))) for s, z in enumerate(f.values))
    components, confidence, edges = _assemble_components(c, f, edges, faces, refs, set(), .95)
    return BandCoverMesh(c, f, tuple(edges), tuple(faces), components, confidence,
                        diagnostics={'independent_classification': info or {}})
