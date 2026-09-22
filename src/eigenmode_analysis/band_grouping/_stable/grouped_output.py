"""Pure raw-grid geometry for Stable v1 (no consumer payload interpretation).

Component ordering is the frozen exporter ordering with no truncation. Only
original candidates form vertices; only complete original cells form faces.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RawGrouping:
    rows: tuple
    height_real: np.ndarray
    height_imag: np.ndarray
    source_rows: np.ndarray
    triangles: tuple


def build_grouped_output(mesh):
    if mesh.branch_cut_edges:
        raise ValueError('ordinary grouping cannot encode explicit branch cuts')
    c, f = mesh.complex, mesh.fibers
    if c.grid_shape is None or any(len(face) != 4 for face in c.faces):
        raise ValueError('ordinary original-grid geometry required')
    groups = [component.vertices for component in mesh.components]
    known = set(f.refs())
    seen = set()
    for group in groups:
        if len({r.site for r in group}) != len(group):
            raise ValueError('one band contains multiple candidates at one site')
        for ref in group:
            if ref not in known or ref in seen:
                raise ValueError('grouping repeats or invents an observation')
            seen.add(ref)
    if seen != known:
        raise ValueError('grouping omits observations')
    def order(gid):
        refs = groups[gid]
        first = min(r.site for r in refs)
        values = sorted((f.value(r).real, f.value(r).imag, r.local) for r in refs if r.site == first)
        return first, values[0], gid
    ordered = sorted(range(len(groups)), key=order)
    labels = {ref: band for band, gid in enumerate(ordered) for ref in groups[gid]}
    shape = (len(groups), *c.grid_shape)
    real = np.full(shape, np.nan)
    imag = np.full(shape, np.nan)
    source = np.full(shape, -1, dtype=int)
    rows = []
    for index, ref in enumerate(f.refs()):
        band = labels[ref]
        value = f.value(ref)
        position = (band, *c.grid_index(ref.site))
        real[position], imag[position], source[position] = value.real, value.imag, index
        rows.append({'source_row': index, 'site': ref.site, 'local': ref.local,
            'original_index': int(f.original_indices[ref.site][ref.local]),
            'band_id': band, 'xy': c.coordinates[ref.site].tolist(),
            'frequency_real': value.real, 'frequency_imag': value.imag})
    triangles = []
    for fid, sites in enumerate(c.faces):
        corners = np.stack([source[(slice(None), *c.grid_index(s))] for s in sites], axis=1)
        for band in np.flatnonzero(np.all(corners >= 0, axis=1)):
            for a, b, d in ((0, 1, 2), (0, 2, 3)):
                triangles.append({'vertices': [int(corners[band, i]) for i in (a, b, d)],
                    'band_id': int(band), 'base_face': fid})
    for array in (real, imag, source):
        array.setflags(write=False)
    return RawGrouping(tuple(rows), real, imag, source, tuple(triangles))
