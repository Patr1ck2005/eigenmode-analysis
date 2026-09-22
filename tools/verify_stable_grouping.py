"""Explicit consumer characterization against the frozen local Stable v1 ZIP.

Not imported by the package. Requires explicit trust before loading local
pickle data; the entire bundle and selected entries must match pinned hashes.
Inputs contain no grouping labels. Expected output is opened only AFTER each
fresh computation. Reports are create-only and preserve partial failure state.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import zipfile
import numpy as np
from numpy.testing import assert_equal

from eigenmode_analysis.band_grouping import (
    ALGORITHM_VERSION, BandCandidates, GroupingConfig, ParameterGrid, group_bands,
)
from eigenmode_analysis.runtime import runtime_info

ARCHIVE_SHA256 = '794bf73f4cca56184f95575a5d493ac56822611c207820a32f92faba9709d67c'
CASES = ('projection_crossing', 'research_avoided_matrix_76001_n25',
         'hard_ep2_critical', 'projection_true_hole',
         'ok_7_c6v_casekb--window-1-h4', 'chiral_eta0.225--window-1-h4')


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def compare_output(result, expected):
    rows = expected['rows']
    assert len(rows) == result.n_candidates
    labels = [np.full(len(z), -1, int) for z in result.candidates.values]
    for index, row in enumerate(rows):
        s, j, b = row['site'], row['local'], row['band_id']
        value = result.candidates.values[s][j]
        assert row['source_row'] == index
        assert row['frequency_real'] == value.real and row['frequency_imag'] == value.imag
        assert row['xy'] == result.grid.coordinates[s].tolist()
        assert row['original_index'] == result.candidates.original_indices[s][j]
        labels[s][j] = b
    for actual, wanted in zip(result.band_ids, labels):
        assert_equal(actual, wanted)
    source = np.asarray(expected['source_rows'], dtype=object).copy()
    source[source == None] = -1  # JSON null is genuine absence, not a row.
    assert_equal(result.source_rows, source.astype(int))
    for key in ('height_real', 'height_imag'):
        assert_equal(getattr(result, key), np.asarray(expected[key], float))
    assert_equal(result.triangles, np.asarray([t['vertices'] for t in expected['triangles']], int).reshape(-1, 3))
    assert_equal(result.triangle_band_ids, [t['band_id'] for t in expected['triangles']])
    assert_equal(result.triangle_cells, [t['base_face'] for t in expected['triangles']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--trust-local-bundle', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-wheel', action='store_true')
    args = parser.parse_args()
    if not args.trust_local_bundle:
        parser.error('pickle requires --trust-local-bundle; do not accept unknown bundles')
    if args.output.exists():
        raise FileExistsError('verification output already exists; inspect it instead of rerunning')
    if hashlib.sha256(args.bundle.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('bundle does not match the frozen Stable v1 release')
    provenance = runtime_info().to_dict()
    if args.require_wheel and provenance['install_mode'] not in ('wheel', 'installed'):
        raise RuntimeError('expected installed wheel, not editable/source-tree')
    args.output.mkdir(parents=True)
    import eigenmode_analysis.band_grouping as module
    source_root = Path(module.__file__).parent
    report = {'algorithm': ALGORITHM_VERSION, 'bundle_sha256': ARCHIVE_SHA256,
        'runtime': provenance, 'config': asdict(GroupingConfig()), 'status': 'running',
        'runtime_source_sha256': {p.relative_to(source_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(source_root.rglob('*.py'))},
        'cases': [], 'uses_historical_labels_as_input': False,
        'physical_qualification_claimed': False, 'scope': 'frozen daily6 exact migration equivalence'}
    write(args.output/'report.json', report)
    try:
        with zipfile.ZipFile(args.bundle) as archive:
            manifest = json.loads(archive.read('MANIFEST.json'))
            def checked(name):
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != manifest['files'][name]:
                    raise ValueError('entry checksum mismatch: '+name)
                return data
            for case in CASES:
                print('START', case, flush=True)
                input_bytes = checked('inputs/'+case+'.pkl')
                raw = pickle.loads(input_bytes)
                assert not {'assignment', 'rows', 'band_id', 'band_ids'}.intersection(raw)
                grid = ParameterGrid(raw['coordinates'], tuple(map(tuple, raw['edges'])),
                    tuple(map(tuple, raw['faces'])), tuple(raw['grid_shape']),
                    tuple(raw['axis_names']), tuple(raw['axis_values']))
                data = BandCandidates(tuple(raw['values']), tuple(raw['original_indices']),
                    tuple(raw['payload']), raw['descriptors'])
                tick = time.perf_counter()
                result = group_bands(grid, data)
                seconds = time.perf_counter()-tick
                # Freeze numeric result before loading any prior classification.
                np.savez_compressed(args.output/(case+'.npz'), assignment=result.assignment,
                    band_ids=np.concatenate(result.band_ids), offsets=data.offsets,
                    height_real=result.height_real, height_imag=result.height_imag,
                    source_rows=result.source_rows, triangles=result.triangles,
                    triangle_band_ids=result.triangle_band_ids, triangle_cells=result.triangle_cells)
                expected = json.loads(checked('outputs/'+case+'/bands.json'))
                compare_output(result, expected)
                for name in ('values','original_indices','payload','descriptors'):
                    actual = dict(data.descriptors) if name == 'descriptors' else getattr(data, name)
                    assert_equal(actual, raw[name])
                row = {'case': case, 'status': 'exact_match', 'seconds': seconds,
                    'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
                    'observations': result.n_candidates, 'bands': result.n_bands,
                    'missing_slots': int(np.count_nonzero(result.source_rows < 0)),
                    'triangles': len(result.triangles), 'all_labels_and_geometry_equal': True,
                    'all_original_fields_unchanged': True,
                    'factor_status': [b.get('factor_status_counts', {}) for b in result.diagnostics['blocks']]}
                report['cases'].append(row)
                write(args.output/'report.json', report)
                print('MATCH', case, row['observations'], row['bands'], row['triangles'], round(seconds, 2), flush=True)
        report['status'] = 'passed'
        report['all_six_equal'] = len(report['cases']) == len(CASES)
        write(args.output/'report.json', report)
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        write(args.output/'report.json', report)
        raise
    print('VERIFIED', len(report['cases']), 'fresh shared computations; frozen labels are evaluation only', flush=True)


if __name__ == '__main__':
    main()
