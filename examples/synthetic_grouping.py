"""Independent 0.3.1 grouping example: synthetic observations, CSVs and plain SVG.

Run in a separate 0.3.1 environment:
    python examples/synthetic_grouping.py --out .local/advanced-grouping
Only NumPy, SciPy and the standard library are needed. No consumer is imported.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
import hashlib
import html
from importlib import metadata
import json
from pathlib import Path
import sys

import eigenmode_analysis
import numpy as np
from eigenmode_analysis.band_grouping import (
    BandCandidates, GroupingConfig, ParameterGrid, group_bands,
)

PUBLIC_SOURCE = "d82251086f50cb1428c35f457631f74c10b5afc2"
SEED = 1729


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_value(value):
    """Serialize diagnostics without silently turning nonfinite values into zero."""
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, complex):
        return {"real": json_value(value.real), "imag": json_value(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": str(value)}
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def example_input():
    u, v = np.linspace(-1., 1., 7), np.linspace(-.7, .7, 7)
    grid = ParameterGrid.regular_grid(u, v, axis_names=("u", "v"))
    rng = np.random.default_rng(SEED)
    missing = grid.site_index((3, 3))
    values, original, payload = [], [], []
    for site, (x, y) in enumerate(grid.coordinates):
        # These are dimensionless mathematical observations, not solver modes.
        z = np.array([1 + .25*x + .03*y + .02j,
                      1 - .25*x + .03*y + .02j])
        order = np.array([], dtype=int) if site == missing else rng.permutation(2)
        values.append(z[order].copy())
        original.append(2*site + order)
        payload.append(tuple(f"synthetic-{2*site+j:03d}" for j in order))
    return grid, BandCandidates(tuple(values), tuple(original), tuple(payload)), missing


def svg_preview(grid, candidates, result, missing):
    """Two ordinary 2D views; no interpolation, surface, or physical truth labels."""
    colors = ["#1257df", "#e31a35", "#009b83", "#8d39c7", "#e18400"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="105mm" height="68mm" viewBox="0 0 397 257">',
             '<rect width="397" height="257" fill="white"/>',
             '<g font-family="Arial,Helvetica,sans-serif" font-size="10.6667" fill="#172238">']

    def text(x, y, value, extra=""):
        parts.append(f'<text x="{x:g}" y="{y:g}" {extra}>{html.escape(str(value))}</text>')

    def line(x1, y1, x2, y2, color="#172238", width=.67):
        parts.append(f'<path d="M{x1:g},{y1:g} L{x2:g},{y2:g}" fill="none" stroke="{color}" stroke-width="{width:g}"/>')

    def axes(left, bottom, width, height, ymin, ymax, ylabel, panel):
        def xy(x, y):
            return left + (x+1)*width/2, bottom - (y-ymin)*height/(ymax-ymin)
        line(left, bottom, left+width, bottom)
        line(left, bottom, left, bottom-height)
        for x in (-1, 0, 1):
            xx, yy = xy(x, ymin)
            line(xx, yy, xx, yy-3)
            text(xx, yy+14, x, 'text-anchor="middle"')
        for y in (ymin, (ymin+ymax)/2, ymax):
            xx, yy = xy(-1, y)
            line(xx, yy, xx+3, yy)
            text(xx-5, yy+3.5, f"{y:g}", 'text-anchor="end"')
        text(left+width/2, bottom+29, "u", 'text-anchor="middle"')
        text(left-29, bottom-height/2, ylabel,
             f'text-anchor="middle" transform="rotate(-90 {left-29:g} {bottom-height/2:g})"')
        text(left-20, bottom-height-10, panel, 'font-size="12"')
        return xy

    xy = axes(38, 161, 129, 129, -.7, .7, "v", "(a)")
    for site, (x, y) in enumerate(grid.coordinates):
        px, py = xy(x, y)
        if site == missing:
            parts.append(f'<circle cx="{px:g}" cy="{py:g}" r="3.3" fill="white" stroke="#e31a35" stroke-width="1.2"/>')
            line(px-2, py-2, px+2, py+2, "#e31a35", 1)
            line(px-2, py+2, px+2, py-2, "#e31a35", 1)
        else:
            parts.append(f'<circle cx="{px:g}" cy="{py:g}" r="2.5" fill="#1257df"/>')
    text(38, 207, "Filled: 2 raw observations", 'font-size="9.3333"')
    text(38, 220, "Cross: 0 observations", 'font-size="9.3333"')

    xy = axes(239, 161, 141, 129, .7, 1.3, "Re z", "(b)")
    # Select an observed v row: the center row contains the genuinely empty site.
    section = 2
    for i in range(7):
        site = grid.site_index((i, section))
        for local, z in enumerate(candidates.values[site]):
            band = int(result.band_ids[site][local])
            px, py = xy(grid.coordinates[site, 0], z.real)
            color = colors[band % len(colors)]
            if band % 2:
                line(px-3.2, py-3.2, px+3.2, py+3.2, color, 1.1)
                line(px-3.2, py+3.2, px+3.2, py-3.2, color, 1.1)
            else:
                parts.append(f'<circle cx="{px:g}" cy="{py:g}" r="3.5" fill="none" stroke="{color}" stroke-width="1.1"/>')
    for band in range(result.n_bands):
        px, py = 247 + 64*band, 201
        color = colors[band % len(colors)]
        if band % 2:
            line(px-3, py-3, px+3, py+3, color, 1.1)
            line(px-3, py+3, px+3, py-3, color, 1.1)
        else:
            parts.append(f'<circle cx="{px:g}" cy="{py:g}" r="3.5" fill="none" stroke="{color}" stroke-width="1.1"/>')
        text(px+8, py+3, f"Group {band}", 'font-size="9.3333"')
    text(239, 217, f"v = {grid.axis_values[1][section]:.3f}; no fitted points", 'font-size="9.3333"')
    text(239, 230, "Equal observations overlap at u = 0", 'font-size="9.3333"')
    text(198.5, 250, "Dimensionless synthetic values; colors/markers denote numerical group IDs.",
         'text-anchor="middle" font-size="9.3333"')
    parts.extend(["</g>", "</svg>"])
    return "\n".join(parts) + "\n"


def main(out):
    if metadata.version("eigenmode-analysis") != "0.3.1":
        raise RuntimeError("Use an isolated eigenmode-analysis 0.3.1 environment.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    targets = ["raw-observations.csv", "grouped-observations.csv", "assignment.csv",
               "triangles.csv", "diagnostics.json", "grouping.svg", "manifest.json"]
    if any((out / name).exists() for name in targets):
        raise FileExistsError("Choose a fresh --out directory; existing receipts are preserved.")
    grid, candidates, missing = example_input()
    before = tuple(z.tobytes() for z in candidates.values)
    original_before = tuple(z.tobytes() for z in candidates.original_indices)
    payload_before = candidates.payload
    raw_rows = [(s, j, int(candidates.original_indices[s][j]), candidates.payload[s][j],
                 *grid.coordinates[s], z.real, z.imag)
                for s, values in enumerate(candidates.values) for j, z in enumerate(values)]
    fields = ["site", "local_index", "original_id", "raw_identity", "u", "v", "real", "imag"]
    raw_path = out / "raw-observations.csv"
    write_csv(raw_path, fields, raw_rows)
    raw_sha = sha(raw_path.read_bytes())
    result = group_bands(grid, candidates, config=GroupingConfig())
    assert result.candidates is candidates and result.grid is grid
    assert before == tuple(z.tobytes() for z in candidates.values)
    assert original_before == tuple(z.tobytes() for z in candidates.original_indices)
    assert payload_before == result.candidates.payload
    assert result.n_candidates == len(raw_rows) == 96
    grouped = []
    for site, z in enumerate(candidates.values):
        selected = result.assignment[site][result.assignment[site] >= 0]
        assert sorted(selected.tolist()) == list(range(len(z)))
        assert len(set(result.band_ids[site].tolist())) == len(z)
        for local, value in enumerate(z):
            band = int(result.band_ids[site][local])
            assert result.assignment[site, band] == local
            index = (band, *grid.grid_index(site))
            assert result.height_real[index] == value.real
            assert result.height_imag[index] == value.imag
            assert result.source_rows[index] == candidates.offsets[site] + local
            grouped.append((*raw_rows[candidates.offsets[site]+local], band))
    assert np.all(result.assignment[missing] == -1)
    assert np.all(result.source_rows[:, 3, 3] == -1)
    assert np.isnan(result.height_real[:, 3, 3]).all()
    equal_sites = [s for s, z in enumerate(candidates.values) if len(z) == 2 and z[0] == z[1]]
    assert len(equal_sites) == 6
    assert all(len(set(result.band_ids[s].tolist())) == 2 for s in equal_sites)
    write_csv(out / "grouped-observations.csv", fields+["band_id"], grouped)
    assert [row[:-1] for row in grouped] == raw_rows
    write_csv(out / "assignment.csv", ["site", "u", "v"]+[f"band_{b}_local" for b in range(result.n_bands)],
              [(s, *grid.coordinates[s], *result.assignment[s]) for s in range(grid.n_sites)])
    triangles = []
    for vertices, band, cell in zip(result.triangles, result.triangle_band_ids, result.triangle_cells):
        assert all(result.assignment[s, band] >= 0 for s in grid.faces[cell])
        assert all(0 <= row < len(raw_rows) for row in vertices)
        assert all(grouped[row][-1] == band for row in vertices)
        triangles.append((int(cell), int(band), *vertices))
    write_csv(out / "triangles.csv", ["cell", "band_id", "raw_row_0", "raw_row_1", "raw_row_2"], triangles)
    assert raw_sha == sha(raw_path.read_bytes())
    assert result.diagnostics["generated_observations"] == 0
    assert result.diagnostics["probability"] is None
    write_json(out / "diagnostics.json", result.diagnostics)
    (out / "grouping.svg").write_text(svg_preview(grid, candidates, result, missing), encoding="utf-8")
    source_root = Path(eigenmode_analysis.__file__).parent
    source_hashes = {p.relative_to(source_root).as_posix():sha(p.read_bytes())
                     for p in sorted(source_root.rglob("*.py"))}
    checks = {"raw_observations":96, "assigned_observations":len(grouped), "empty_sites":1,
        "equal_frequency_sites":len(equal_sites), "equal_frequency_observations":2*len(equal_sites),
        "source_value_bytes_unchanged":True, "raw_csv_bytes_unchanged":True,
        "original_indices_and_payload_unchanged":True, "each_raw_identity_exactly_once":True,
        "all_exported_values_equal_original_values":True, "empty_site_remains_missing":True,
        "triangles_use_only_original_rows":True, "all_assertions_passed":True}
    manifest = {"input_kind":"independently defined dimensionless mathematics", "public_api_source":PUBLIC_SOURCE,
        "script_sha256":sha(Path(__file__).read_bytes()), "seed":SEED, "grid_shape":grid.grid_shape,
        "axes": {"u":grid.axis_values[0], "v":grid.axis_values[1]}, "missing_grid_index":[3,3],
        "definition":"z = 1 +/- 0.25*u + 0.03*v + 0.02j; two observations per observed site, shuffled independently",
        "versions":{"python":sys.version.split()[0], "numpy":np.__version__, "scipy":metadata.version("scipy"),
                    "eigenmode-analysis":metadata.version("eigenmode-analysis"), "algorithm":result.algorithm_version},
        "installed_python_sources":source_hashes, "groups":result.n_bands, "triangles":len(triangles),
        "checks":checks, "raw_csv_sha256":raw_sha,
        "diagnostic_status": {k:result.diagnostics[k] for k in ("software_status", "scientific_status", "physical_correctness_proven", "probability")},
        "scope":"Observation preservation demonstration; group labels are not certified physical identities or calibrated probabilities.",
        "figure":{"width_mm":105, "height_mm":68, "font":"Arial with Helvetica/sans-serif fallback", "labels_pt":8, "notes_pt":7,
                  "transformation":"2D support plus observed constant-v section; scatter only, no fitted/interpolated observations"},
        "outputs":{name:sha((out/name).read_bytes()) for name in targets if name != "manifest.json"}}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"groups":result.n_bands, "checks":checks, "diagnostic_status":manifest["diagnostic_status"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".local/advanced-grouping"))
    main(parser.parse_args().out)
