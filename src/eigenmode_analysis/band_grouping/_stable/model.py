"""Typed representation of a band cover over a parameter cell complex.

The key invariant is provenance: every cover vertex is a reference to one
input candidate.  Tracking may connect, split into components, or mark a
choice ambiguous, but it never manufactures a frequency sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _readonly(array: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    out = np.asarray(array, dtype=dtype).copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True, order=True)
class CandidateRef:
    """A candidate's stable address: ``(parameter site, local index)``."""

    site: int
    local: int


@dataclass(frozen=True)
class ParameterCellComplex:
    """Coordinates, graph edges, and optional oriented 2-cells.

    ``grid_shape`` and ``axis_values`` are adapter metadata.  The algorithms
    consume only coordinates/edges/faces, so an explicitly supplied
    irregular complex can use the same type in a later phase.
    """

    coordinates: np.ndarray
    edges: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, ...], ...] = ()
    grid_shape: tuple[int, ...] | None = None
    axis_names: tuple[str, ...] = ()
    axis_values: tuple[np.ndarray, ...] = ()
    _edge_lookup: Mapping[tuple[int, int], int] = field(
        init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        coords = _readonly(self.coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[0] == 0:
            raise ValueError("coordinates must have shape (n_sites, dimension)")
        if not np.all(np.isfinite(coords)):
            raise ValueError("parameter coordinates must be finite")

        n_sites = coords.shape[0]
        normalized_edges: list[tuple[int, int]] = []
        for raw in self.edges:
            if len(raw) != 2:
                raise ValueError(f"invalid edge {raw!r}")
            a, b = (int(raw[0]), int(raw[1]))
            if a == b or min(a, b) < 0 or max(a, b) >= n_sites:
                raise ValueError(f"edge outside complex: {(a, b)!r}")
            normalized_edges.append((min(a, b), max(a, b)))
        if len(set(normalized_edges)) != len(normalized_edges):
            raise ValueError("duplicate parameter edges")
        normalized_edges.sort()
        edge_set = set(normalized_edges)

        normalized_faces: list[tuple[int, ...]] = []
        for raw in self.faces:
            face = tuple(int(v) for v in raw)
            if len(face) < 3 or len(set(face)) != len(face):
                raise ValueError(f"invalid face {face!r}")
            if min(face) < 0 or max(face) >= n_sites:
                raise ValueError(f"face outside complex: {face!r}")
            perimeter = zip(face, face[1:] + face[:1])
            if any((min(a, b), max(a, b)) not in edge_set for a, b in perimeter):
                raise ValueError(f"face perimeter is missing an edge: {face!r}")
            normalized_faces.append(face)

        shape = None if self.grid_shape is None else tuple(int(n) for n in self.grid_shape)
        if shape is not None and (np.prod(shape) != n_sites or any(n < 1 for n in shape)):
            raise ValueError("grid_shape does not match coordinates")
        axes = tuple(_readonly(a, dtype=float) for a in self.axis_values)
        if axes and (shape is None or len(axes) != len(shape)):
            raise ValueError("axis_values require one axis per grid dimension")
        if axes and any(len(axis) != shape[i] for i, axis in enumerate(axes)):
            raise ValueError("axis_values lengths do not match grid_shape")

        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "edges", tuple(normalized_edges))
        object.__setattr__(self, "_edge_lookup", MappingProxyType(
            {edge: i for i, edge in enumerate(normalized_edges)}))
        object.__setattr__(self, "faces", tuple(normalized_faces))
        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "axis_values", axes)
        if self.axis_names and len(self.axis_names) != coords.shape[1]:
            raise ValueError("axis_names do not match coordinate dimension")

    @property
    def n_sites(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def coordinate_dimension(self) -> int:
        return int(self.coordinates.shape[1])

    @property
    def topological_dimension(self) -> int:
        return 2 if self.faces else 1

    def edge_index(self) -> Mapping[tuple[int, int], int]:
        return self._edge_lookup

    def site_index(self, grid_index: Sequence[int]) -> int:
        if self.grid_shape is None:
            raise ValueError("this complex has no regular-grid metadata")
        return int(np.ravel_multi_index(tuple(grid_index), self.grid_shape))

    def grid_index(self, site: int) -> tuple[int, ...]:
        if self.grid_shape is None:
            raise ValueError("this complex has no regular-grid metadata")
        return tuple(int(i) for i in np.unravel_index(site, self.grid_shape))

    @classmethod
    def line(cls, axis: Sequence[float], *, name: str = "x") -> "ParameterCellComplex":
        values = np.asarray(axis, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("line axis must be a non-empty vector")
        edges = tuple((i, i + 1) for i in range(values.size - 1))
        return cls(
            coordinates=values[:, None],
            edges=edges,
            grid_shape=(values.size,),
            axis_names=(name,),
            axis_values=(values,),
        )

    @classmethod
    def regular_grid(
        cls,
        *axes: Sequence[float],
        axis_names: Sequence[str] | None = None,
    ) -> "ParameterCellComplex":
        """Build a 1D chain or 2D quadrilateral cell complex."""
        if len(axes) == 1:
            name = axis_names[0] if axis_names else "x"
            return cls.line(axes[0], name=name)
        if len(axes) != 2:
            raise ValueError("the first release supports regular 1D/2D grids")
        av = tuple(np.asarray(a, dtype=float) for a in axes)
        if any(a.ndim != 1 or a.size == 0 for a in av):
            raise ValueError("each grid axis must be a non-empty vector")
        shape = (av[0].size, av[1].size)
        mesh = np.meshgrid(*av, indexing="ij")
        coords = np.stack([m.reshape(-1) for m in mesh], axis=1)
        edges: list[tuple[int, int]] = []
        faces: list[tuple[int, ...]] = []
        sid = lambda i, j: int(np.ravel_multi_index((i, j), shape))
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i + 1 < shape[0]:
                    edges.append((sid(i, j), sid(i + 1, j)))
                if j + 1 < shape[1]:
                    edges.append((sid(i, j), sid(i, j + 1)))
                if i + 1 < shape[0] and j + 1 < shape[1]:
                    faces.append((sid(i, j), sid(i + 1, j), sid(i + 1, j + 1), sid(i, j + 1)))
        names = tuple(axis_names) if axis_names else ("x", "y")
        return cls(
            coordinates=coords,
            edges=tuple(edges),
            faces=tuple(faces),
            grid_shape=shape,
            axis_names=names,
            axis_values=av,
        )


@dataclass(frozen=True)
class CandidateFibers:
    """Variable-cardinality unordered complex-frequency fibers."""

    values: tuple[np.ndarray, ...]
    original_indices: tuple[np.ndarray, ...] = ()
    payload: tuple[tuple[Any, ...], ...] = ()
    descriptors: Mapping[str, tuple[np.ndarray, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vals = tuple(_readonly(np.asarray(v).reshape(-1), dtype=complex) for v in self.values)
        if not vals:
            raise ValueError("at least one candidate fiber is required")
        if any(not np.all(np.isfinite(v)) for v in vals):
            raise ValueError("candidate values must be finite; screen failed samples first")

        if self.original_indices:
            original = tuple(_readonly(np.asarray(v).reshape(-1), dtype=int)
                             for v in self.original_indices)
        else:
            original = tuple(_readonly(np.arange(len(v)), dtype=int) for v in vals)
        if len(original) != len(vals) or any(len(a) != len(v) for a, v in zip(original, vals)):
            raise ValueError("original_indices do not align with fibers")
        if any(len(set(a.tolist())) != len(a) for a in original):
            raise ValueError("original candidate indices must be unique within each fiber")

        if self.payload:
            payload = tuple(tuple(cell) for cell in self.payload)
        else:
            payload = tuple(tuple(None for _ in v) for v in vals)
        if len(payload) != len(vals) or any(len(a) != len(v) for a, v in zip(payload, vals)):
            raise ValueError("payload does not align with fibers")

        desc: dict[str, tuple[np.ndarray, ...]] = {}
        for name, arrays in self.descriptors.items():
            packed = tuple(_readonly(np.asarray(a).reshape(-1), dtype=float) for a in arrays)
            if len(packed) != len(vals) or any(len(a) != len(v) for a, v in zip(packed, vals)):
                raise ValueError(f"descriptor {name!r} does not align with fibers")
            desc[str(name)] = packed

        object.__setattr__(self, "values", vals)
        object.__setattr__(self, "original_indices", original)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "descriptors", MappingProxyType(desc))

    @property
    def n_sites(self) -> int:
        return len(self.values)

    @property
    def n_candidates(self) -> int:
        return int(sum(len(v) for v in self.values))

    @property
    def offsets(self) -> np.ndarray:
        return np.cumsum([0] + [len(v) for v in self.values], dtype=int)

    def refs(self) -> Iterable[CandidateRef]:
        for site, values in enumerate(self.values):
            for local in range(len(values)):
                yield CandidateRef(site, local)

    def value(self, ref: CandidateRef) -> complex:
        return complex(self.values[ref.site][ref.local])

    def payload_for(self, ref: CandidateRef) -> Any:
        return self.payload[ref.site][ref.local]

    def global_id(self, ref: CandidateRef) -> int:
        offsets = self.offsets
        return int(offsets[ref.site] + ref.local)


@dataclass(frozen=True)
class AssociationEdge:
    """One MAP association plus *additional* prediction-set alternatives."""

    base_edge: int
    source: CandidateRef
    target: CandidateRef
    cost: float
    confidence: float
    committed: bool
    alternatives: tuple[CandidateRef, ...] = ()
    unmatched_alternative: bool = False


@dataclass(frozen=True)
class CoverFace:
    """One lifted face (a sheet patch) above a parameter-space face."""

    base_face: int
    vertices: tuple[CandidateRef, ...]
    confidence: float
    committed: bool


@dataclass(frozen=True)
class CoverComponent:
    """Connected component in the lifted candidate graph."""

    component_id: int
    vertices: tuple[CandidateRef, ...]
    edge_indices: tuple[int, ...]
    face_indices: tuple[int, ...]
    confidence: float
    status: str  # committed | ambiguous | fragment | isolated


@dataclass(frozen=True)
class BranchDefect:
    """Localized non-trivial holonomy retained by the covering model."""

    base_faces: tuple[int, ...]
    loop_sites: tuple[int, ...]
    permutation: tuple[int, ...]
    support: float
    confidence: float
    branch_cut_edges: tuple[int, ...]
    evidence: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BandCoverMesh:
    """Reconstructed discrete cover, including ambiguity and defects."""

    complex: ParameterCellComplex
    fibers: CandidateFibers
    edges: tuple[AssociationEdge, ...]
    faces: tuple[CoverFace, ...]
    components: tuple[CoverComponent, ...]
    vertex_confidence: Mapping[CandidateRef, float]
    multiplicity: Mapping[CandidateRef, int] = field(default_factory=dict)
    branch_defects: tuple[BranchDefect, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.complex.n_sites != self.fibers.n_sites:
            raise ValueError("cell complex and fibers have different site counts")
        def known_candidate(ref: CandidateRef) -> bool:
            return (
                0 <= ref.site < self.fibers.n_sites
                and 0 <= ref.local < len(self.fibers.values[ref.site])
            )

        errors: list[str] = []
        for i, edge in enumerate(self.edges):
            if not known_candidate(edge.source) or not known_candidate(edge.target):
                errors.append(f"edge {i} references an unknown candidate")
                continue
            if not (0 <= edge.base_edge < len(self.complex.edges)):
                errors.append(f"edge {i} references an unknown base edge")
                continue
            a, b = self.complex.edges[edge.base_edge]
            if (edge.source.site, edge.target.site) != (a, b):
                errors.append(f"edge {i} orientation does not match its base edge")
            if not (0.0 <= edge.confidence <= 1.0):
                errors.append(f"edge {i} has invalid confidence")
        for i, face in enumerate(self.faces):
            if not (0 <= face.base_face < len(self.complex.faces)):
                errors.append(f"face {i} references an unknown base face")
            elif tuple(v.site for v in face.vertices) != self.complex.faces[face.base_face]:
                errors.append(f"face {i} does not lift its base face in order")
            if len(set(face.vertices)) != len(face.vertices):
                errors.append(f"face {i} repeats a candidate")
        if any(not known_candidate(ref) or int(m) < 1
               for ref, m in self.multiplicity.items()):
            errors.append("invalid explicit multiplicity")
        if errors:
            raise ValueError("invalid BandCoverMesh: " + "; ".join(errors))

        vertex_conf = {ref: float(value) for ref, value in self.vertex_confidence.items()}
        if (len(vertex_conf) != self.fibers.n_candidates
                or any(not known_candidate(ref) for ref in vertex_conf)):
            raise ValueError("vertex_confidence must cover every input candidate")
        if any(not 0.0 <= value <= 1.0 for value in vertex_conf.values()):
            raise ValueError("invalid vertex confidence")
        object.__setattr__(self, "vertex_confidence", MappingProxyType(vertex_conf))
        object.__setattr__(self, "multiplicity", MappingProxyType(
            {ref: int(m) for ref, m in self.multiplicity.items()}))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def branch_cut_edges(self) -> tuple[int, ...]:
        return tuple(sorted({edge for defect in self.branch_defects
                             for edge in defect.branch_cut_edges}))

    def validation_errors(self) -> list[str]:
        """Return matching violations that are meaningful to evaluators."""
        errors: list[str] = []
        per_base: dict[int, tuple[set[CandidateRef], set[CandidateRef]]] = {}
        forward: dict[tuple[int, CandidateRef], CandidateRef] = {}
        backward: dict[tuple[int, CandidateRef], CandidateRef] = {}
        for i, edge in enumerate(self.edges):
            sources, targets = per_base.setdefault(edge.base_edge, (set(), set()))
            if edge.source in sources and self.multiplicity.get(edge.source, 1) == 1:
                errors.append(f"duplicate source candidate on base edge {edge.base_edge}: {edge.source}")
            if edge.target in targets and self.multiplicity.get(edge.target, 1) == 1:
                errors.append(f"duplicate target candidate on base edge {edge.base_edge}: {edge.target}")
            sources.add(edge.source)
            targets.add(edge.target)
            forward[(edge.base_edge, edge.source)] = edge.target
            backward[(edge.base_edge, edge.target)] = edge.source
            if edge.committed and edge.alternatives:
                errors.append(f"committed edge {i} has alternatives")

        def cycle_signature(permutation: Sequence[int]) -> tuple[int, ...]:
            seen: set[int] = set()
            cycles: list[int] = []
            for start in range(len(permutation)):
                if start in seen or permutation[start] < 0:
                    continue
                current = start
                size = 0
                while (current not in seen
                       and 0 <= current < len(permutation)):
                    seen.add(current)
                    size += 1
                    current = int(permutation[current])
                if current == start and size > 1:
                    cycles.append(size)
            return tuple(sorted(cycles))

        edge_lookup = self.complex.edge_index()
        for defect_id, defect in enumerate(self.branch_defects):
            if (not defect.loop_sites or len(defect.loop_sites) < 2
                    or defect.loop_sites[0] != defect.loop_sites[-1]):
                errors.append(f"branch defect {defect_id} has no closed loop")
                continue
            if any(not 0 <= face_id < len(self.complex.faces)
                   for face_id in defect.base_faces):
                errors.append(f"branch defect {defect_id} references an unknown face")
            if any(not 0 <= edge_id < len(self.complex.edges)
                   for edge_id in defect.branch_cut_edges):
                errors.append(f"branch defect {defect_id} references an unknown cut edge")
            if not (0.0 <= defect.support <= 1.0
                    and 0.0 <= defect.confidence <= 1.0):
                errors.append(f"branch defect {defect_id} has invalid confidence")

            observed: list[int] = []
            for local in range(len(self.fibers.values[defect.loop_sites[0]])):
                current: CandidateRef | None = CandidateRef(
                    defect.loop_sites[0], local)
                for source_site, target_site in zip(
                        defect.loop_sites, defect.loop_sites[1:]):
                    if current is None:
                        break
                    edge_id = edge_lookup.get((
                        min(source_site, target_site),
                        max(source_site, target_site),
                    ))
                    if edge_id is None:
                        current = None
                        break
                    current = (
                        forward.get((edge_id, current))
                        if source_site < target_site
                        else backward.get((edge_id, current))
                    )
                observed.append(-1 if current is None else current.local)
            declared_signature = cycle_signature(defect.permutation)
            observed_signature = cycle_signature(observed)
            if (not declared_signature
                    or observed_signature != declared_signature):
                errors.append(
                    "branch defect "
                    f"{defect_id} transport mismatch: "
                    f"declared {declared_signature}, observed {observed_signature}"
                )
        return errors
