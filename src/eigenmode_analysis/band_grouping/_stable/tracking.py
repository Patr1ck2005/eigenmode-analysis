"""Numerical subset of frozen Stable v1; see migration provenance."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .model import AssociationEdge, BandCoverMesh, CandidateFibers, CandidateRef, CoverComponent, ParameterCellComplex


class _UnionFind:
    """Compact disjoint sets for global candidate ids."""

    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        root = int(item)
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _component_extent(complex_: ParameterCellComplex, vertices: list[CandidateRef]) -> float:
    if not vertices:
        return 0.0
    if complex_.grid_shape is not None:
        indices = np.asarray([complex_.grid_index(v.site) for v in vertices], dtype=int)
        fractions = []
        for axis, length in enumerate(complex_.grid_shape):
            occupied_span = int(indices[:, axis].max() - indices[:, axis].min() + 1)
            fractions.append(occupied_span / length)
        return float(np.prod(fractions))
    coords = complex_.coordinates[[v.site for v in vertices]]
    full_span = np.ptp(complex_.coordinates, axis=0)
    local_span = np.ptp(coords, axis=0)
    active = full_span > 0
    return float(np.prod(local_span[active] / full_span[active])) if np.any(active) else 0.0


def _component_support_cut(extents: list[float]) -> float:
    """Data-derived split between macroscopic sheets and short fragments.

    A component need not cover 95% of the domain: births, deaths, branch cuts,
    and symmetry-point degeneracies can create legitimate half-domain sheets.
    We instead fit the optimal deterministic two-class partition to log extent
    and use its geometric boundary.  This changes confidence only, never MAP.
    """
    values = np.asarray([value for value in extents if value > 0], dtype=float)
    if values.size == 0:
        return 1.0
    values.sort()
    if values.size == 1 or np.isclose(values[0], values[-1], rtol=1e-12, atol=0.0):
        return float(values[0])
    transformed = np.log(np.maximum(values, np.finfo(float).tiny))
    best_split = 1
    best_loss = np.inf
    for split in range(1, len(transformed)):
        left = transformed[:split]
        right = transformed[split:]
        loss = float(
            np.sum((left - np.mean(left)) ** 2)
            + np.sum((right - np.mean(right)) ** 2)
        )
        if loss < best_loss:
            best_loss = loss
            best_split = split
    return float(np.sqrt(values[best_split - 1] * values[best_split]))


def _assemble_components(
    complex_: ParameterCellComplex,
    fibers: CandidateFibers,
    edges: list[AssociationEdge],
    faces,
    candidate_refs: tuple[tuple[CandidateRef, ...], ...],
    branch_cut_edges: set[int],
    confidence_level: float,
) -> tuple[tuple[CoverComponent, ...], dict[CandidateRef, float], list[AssociationEdge]]:
    offsets = fibers.offsets
    union = _UnionFind(fibers.n_candidates)

    def global_id(ref: CandidateRef) -> int:
        return int(offsets[ref.site] + ref.local)

    for edge in edges:
        if edge.base_edge not in branch_cut_edges:
            union.union(global_id(edge.source), global_id(edge.target))
    roots = np.fromiter(
        (union.find(candidate) for candidate in range(fibers.n_candidates)),
        dtype=np.int64,
        count=fibers.n_candidates,
    )
    unique_roots = np.unique(roots)
    component_by_root = {
        int(root): component_id
        for component_id, root in enumerate(unique_roots)
    }
    component_ids = np.fromiter(
        (component_by_root[int(root)] for root in roots),
        dtype=np.int64,
        count=len(roots),
    )
    ordered: list[list[CandidateRef]] = [list() for _ in unique_roots]
    for site_refs in candidate_refs:
        for ref in site_refs:
            ordered[int(component_ids[global_id(ref)])].append(ref)

    def component_of(ref: CandidateRef) -> int:
        return int(component_ids[global_id(ref)])

    edge_ids: dict[int, list[int]] = defaultdict(list)
    for i, edge in enumerate(edges):
        source_component = component_of(edge.source)
        if (edge.base_edge not in branch_cut_edges
                and source_component == component_of(edge.target)):
            edge_ids[source_component].append(i)
    face_ids: dict[int, list[int]] = defaultdict(list)
    for i, face in enumerate(faces):
        ids = {component_of(v) for v in face.vertices}
        if len(ids) == 1:
            face_ids[ids.pop()].append(i)

    extents = {
        cid: _component_extent(complex_, group)
        for cid, group in enumerate(ordered)
    }
    support_cut = _component_support_cut([
        extents[cid] for cid in range(len(ordered)) if edge_ids.get(cid)
    ])

    components: list[CoverComponent] = []
    vertex_confidence: dict[CandidateRef, float] = {}
    # ``edges`` is an internal construction list.  Replacing its entries in
    # place avoids retaining a second million-object edge graph at full size.
    adjusted_edges = edges
    for cid, group in enumerate(ordered):
        incident = edge_ids.get(cid, [])
        raw_conf = [edges[i].confidence for i in incident]
        extent = extents[cid]
        supported = bool(incident) and extent >= support_cut
        support_confidence = 1.0 if supported else min(1.0, extent / support_cut)
        confidence = min(
            float(np.median(raw_conf)) if raw_conf else 0.0,
            support_confidence,
        )
        if not incident:
            status = "isolated"
        elif not supported:
            status = "fragment"
        elif confidence >= confidence_level:
            status = "committed"
        else:
            status = "ambiguous"
        components.append(CoverComponent(
            component_id=cid,
            vertices=tuple(sorted(group)),
            edge_indices=tuple(incident),
            face_indices=tuple(face_ids.get(cid, [])),
            confidence=confidence,
            status=status,
        ))
        incident_conf: dict[CandidateRef, list[float]] = defaultdict(list)
        for edge_id in incident:
            edge = edges[edge_id]
            incident_conf[edge.source].append(min(edge.confidence, support_confidence))
            incident_conf[edge.target].append(min(edge.confidence, support_confidence))
        for ref in group:
            vertex_confidence[ref] = max(incident_conf.get(ref, [0.0]))

        # Component dimensional support is evidence for whether a geometrically
        # smooth fragment should be asserted as a main band.  It changes only
        # confidence, never the MAP connection.
        for edge_id in incident:
            edge = adjusted_edges[edge_id]
            edge_conf = min(edge.confidence, support_confidence)
            adjusted_edges[edge_id] = AssociationEdge(
                base_edge=edge.base_edge,
                source=edge.source,
                target=edge.target,
                cost=edge.cost,
                confidence=edge_conf,
                committed=edge_conf >= confidence_level
                and not edge.alternatives
                and not edge.unmatched_alternative,
                alternatives=edge.alternatives,
                unmatched_alternative=edge.unmatched_alternative,
            )
    return tuple(components), vertex_confidence, adjusted_edges
