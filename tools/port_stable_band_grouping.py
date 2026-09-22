"""One-time, create-only extraction from the SHA-pinned local Stable v1 ZIP.

This maintenance tool is not imported or distributed as runtime code. It
copies the explicit numerical closure, not the research harness. No source
file in the frozen archive or consumer is modified.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = '794bf73f4cca56184f95575a5d493ac56822611c207820a32f92faba9709d67c'
A = 'myplots_app/experimental/band_tracking/'
T = 'tests/bandtrack/'


def extract(text: str, names: tuple[str, ...], imports: str) -> str:
    tree = ast.parse(text)
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    lines = text.splitlines(keepends=True)
    parts = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [v.lineno for v in getattr(node, 'decorator_list', [])]) - 1
        parts.append(''.join(lines[start:node.end_lineno]).rstrip())
    return '"""Numerical subset of frozen Stable v1; see migration provenance."""\n' + imports + '\n\n' + '\n\n\n'.join(parts) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('not the authorized immutable Stable v1 archive')
    out = ROOT / 'src/eigenmode_analysis/band_grouping/_stable'
    if out.exists():
        raise FileExistsError('numerical port already exists; do not overwrite reviewed source')
    targets: dict[str, str] = {}
    provenance = []
    with zipfile.ZipFile(args.archive) as archive:
        manifest = json.loads(archive.read('MANIFEST.json'))
        def source(path: str) -> str:
            name = 'code/' + path
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != manifest['files'][name]:
                raise ValueError('source hash mismatch: ' + name)
            return data.decode('utf-8')
        def add(target: str, path: str, names: tuple[str, ...] | None = None, imports: str = '') -> None:
            text = source(path)
            value = text if names is None else extract(text, names, imports)
            value = value.replace('from myplots_app.experimental.band_tracking.', 'from .')
            value = value.replace('from .real_local_structure_probe import design', 'from .polynomial import design')
            if target == 'scalar_model_grouping':
                value = value.replace('from .scalar_grouping import reconstruct_grouping\n', '')
                old = '    if fallback is None:\n        fallback, _ = reconstruct_grouping(c, f)'
                if value.count(old) != 1:
                    raise ValueError('unexpected scalar-model fallback source')
                value = value.replace(old, '    if fallback is None:\n        raise ValueError("internal scalar grouping requires the precomputed transport seed")')
            ast.parse(value)
            targets[target + '.py'] = value
            provenance.append({'target': target + '.py', 'source': 'code/' + path,
                'source_sha256': manifest['files']['code/' + path],
                'symbols': list(names) if names is not None else 'module',
                'port_sha256': hashlib.sha256(value.encode('utf-8')).hexdigest()})
        for name in ('model', 'model_guided_transport', 'scalar_model_grouping',
                     'scalar_models', 'scalar_strip', 'permutation_transport', 'discriminant_transport'):
            add(name, A + name + '.py')
        add('tracking', A + 'tracking.py',
            ('_UnionFind', '_component_extent', '_component_support_cut', '_assemble_components'),
            'from __future__ import annotations\nfrom collections import defaultdict\nimport numpy as np\nfrom .model import AssociationEdge, BandCoverMesh, CandidateFibers, CandidateRef, CoverComponent, ParameterCellComplex\n')
        add('classification', A + 'classification.py', ('validate_assignment', 'mesh_from_assignment'),
            'import numpy as np\nfrom .model import AssociationEdge, BandCoverMesh, CandidateRef, CoverFace\nfrom .tracking import _assemble_components\n')
        add('local_pair_stitching', T + 'local_pair_stitching.py', ('Union', 'preferences'),
            'from scipy.optimize import linear_sum_assignment\n')
        add('multiband_blocks', T + 'multiband_blocks.py', ('interaction_blocks',),
            'from collections import defaultdict\nfrom .local_pair_stitching import Union\n')
        add('polynomial', T + 'real_local_structure_probe.py', ('design',), 'import numpy as np\n')
        add('projected_connections', T + 'projected_connections.py',
            ('restore_model', 'contact_options', 'raw_chord_costs', 'evidence_edge'),
            'import numpy as np\nfrom .local_spectral_contact import SpectralModel, predict\n')
        for name in ('local_spectral_contact', 'partial_observations', 'partial_observation_grouping', 'stable_v1'):
            add(name, T + name + '.py')
    # Validate all extraction artifacts before the first source write.
    forbidden = {'myplots_app', 'tests', 'matplotlib', 'pandas', 'campaign_data', 'plotly', 'pickle'}
    for name, text in targets.items():
        for node in ast.walk(ast.parse(text)):
            refs = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and not node.level else [])
            if any(v and v.split('.')[0] in forbidden for v in refs):
                raise ValueError('consumer/runtime dependency escaped extraction: ' + name)
    out.mkdir(parents=True)
    for name, text in targets.items():
        (out / name).write_text(text, encoding='utf-8', newline='\n')
    (out / '__init__.py').write_text('"""Private numerical implementation of the frozen Stable v1 policy."""\n', encoding='utf-8')
    record = {'algorithm_version': 'stable-v1.0.0', 'archive_sha256': ARCHIVE_SHA256,
        'source_repository': 'myPlots', 'module_count': len(targets), 'extraction': provenance,
        'changes': ['consumer imports redirected to private sibling modules',
            'unused benchmark/visualization import closure excluded',
            'unreachable standalone G1 fallback removed; G10 always supplies transport fallback',
            'consumer legacy payload exporter replaced by schema-neutral raw geometry']}
    (ROOT / 'docs/stages/phase-12-source-map.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('EXTRACTED', len(targets), 'numerical modules;', sum(len(v.splitlines()) for v in targets.values()), 'lines')


if __name__ == '__main__':
    main()
