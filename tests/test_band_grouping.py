"""Public Stable v1 contract, independent of any consumer or saved result."""
import ast
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from eigenmode_analysis.band_grouping import (
    ALGORITHM_VERSION, BandCandidates, BandGroupingResult, GroupingConfig,
    ParameterGrid, group_bands,
)


def simple(n=3):
    grid = ParameterGrid.regular_grid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    values = tuple(np.array([1 + .03*x + .01j, 2 + .02*y + .04j]) for x, y in grid.coordinates)
    return grid, BandCandidates(values)


def check_result(grid, data, result):
    assert isinstance(result, BandGroupingResult)
    assert result.candidates is data and result.grid is grid
    assert result.n_candidates == data.n_candidates
    for s, z in enumerate(data.values):
        assert sorted(result.assignment[s][result.assignment[s] >= 0]) == list(range(len(z)))
        assert len(set(result.band_ids[s])) == len(z)
        for j, band in enumerate(result.band_ids[s]):
            assert result.assignment[s, band] == j
            idx = (band, *grid.grid_index(s))
            assert result.height_real[idx] == z[j].real
            assert result.height_imag[idx] == z[j].imag
    for array in (result.assignment, result.source_rows, result.height_real,
                  result.height_imag, result.triangles, result.triangle_band_ids, result.triangle_cells):
        assert not array.flags.writeable
    assert result.diagnostics['probability'] is None
    assert result.diagnostics['generated_observations'] == 0
    assert result.diagnostics['deleted_parameter_factors'] == 0
    assert result.algorithm_version == ALGORITHM_VERSION == 'stable-v1.0.0'


def test_fresh_public_api_and_metadata_isolation():
    grid, data = simple()
    altered = BandCandidates(data.values,
        tuple(np.arange(len(z)) + 100 for z in data.values),
        tuple(({'truth': 'wrong'}, {'truth': 'wrong'}) for z in data.values),
        {'not_a_feature': tuple(np.array([-999., 999.]) for z in data.values)})
    first = group_bands(grid, data)
    second = group_bands(grid, altered)
    check_result(grid, data, first)
    check_result(grid, altered, second)
    np.testing.assert_array_equal(first.assignment, second.assignment)
    assert second.candidates.payload == altered.payload
    assert second.candidates.original_indices[0][0] == 100
    assert second.diagnostics['implementation']['distribution'] == 'eigenmode-analysis'


def test_unsupported_small_grid_retains_valid_seed_without_changing_input():
    grid, data = simple()
    seed = np.tile([0, 1], (grid.n_sites, 1)); seed[4] = [1, 0]
    before = seed.copy()
    result = group_bands(grid, data, initial_assignment=seed)
    check_result(grid, data, result)
    np.testing.assert_array_equal(seed, before)
    assert result.diagnostics['blocks'][0]['fallback']
    assert result.diagnostics['blocks'][0]['evidence_status'] == 'resource_or_method_limit'


def test_empty_site_and_equal_frequencies_retain_multiplicity():
    grid, _ = simple()
    data = BandCandidates(tuple(np.array([1+0j, 1+0j]) if i != 4 else np.array([], complex) for i in range(9)))
    seed = np.tile([0, 1], (9, 1)); seed[4] = -1
    result = group_bands(grid, data, initial_assignment=seed)
    check_result(grid, data, result)
    assert result.n_candidates == 16
    assert np.count_nonzero(result.source_rows < 0) == 2
    assert result.triangles.shape == (0, 3)


def test_disconnected_support_is_not_bridged_across_absence():
    grid = ParameterGrid.line([0., 1., 2.])
    data = BandCandidates((np.array([1j]), np.array([], complex), np.array([2j])))
    result = group_bands(grid, data, initial_assignment=np.array([[0], [-1], [0]]))
    check_result(grid, data, result)
    assert result.n_bands == 2
    assert result.band_ids[0][0] != result.band_ids[2][0]
    assert not len(result.triangles)


@pytest.mark.parametrize('bad', [np.zeros((9, 2), int), np.full((9, 2), -2), np.ones((9, 2)), np.zeros((8, 2), int)])
def test_invalid_assignment_is_an_error_not_an_unresolved_case(bad):
    grid, data = simple()
    with pytest.raises(ValueError):
        group_bands(grid, data, initial_assignment=bad)


def test_topology_and_axis_validation():
    grid, data = simple()
    for other in (
        ParameterGrid(grid.coordinates, grid.edges[:-1], (), grid.grid_shape, axis_values=grid.axis_values),
        ParameterGrid(grid.coordinates + 1, grid.edges, grid.faces, grid.grid_shape, axis_values=grid.axis_values),
        ParameterGrid(grid.coordinates, grid.edges, grid.faces, grid.grid_shape),
    ):
        with pytest.raises(ValueError): group_bands(other, data)
    with pytest.raises(ValueError): group_bands(ParameterGrid.regular_grid([1., 0., -1.], [-1., 0., 1.]), data)
    with pytest.raises(ValueError): group_bands(grid, BandCandidates(tuple([] for _ in range(9))))
    with pytest.raises(ValueError): BandCandidates((np.array([np.nan]),))
    with pytest.raises(TypeError): group_bands(grid, data.values)
    with pytest.raises(TypeError): group_bands(grid, data, config={})


def test_one_dimensional_automatic_seed_limit_is_explicit():
    grid = ParameterGrid.line([0., 1., 2.])
    data = BandCandidates(tuple(np.array([1., 2.]) for _ in range(3)))
    with pytest.raises(ValueError, match='automatic scalar seeding is 2D'):
        group_bands(grid, data)


def test_integer_resource_bounds_do_not_silently_round():
    grid, data = simple()
    with pytest.raises(ValueError): group_bands(grid, data, config=GroupingConfig(max_model_roots=2.5))
    with pytest.raises(ValueError): group_bands(grid, data, config=GroupingConfig(max_factors_per_block=True))
    with pytest.raises(ValueError): GroupingConfig(epsilon_f=np.nan)


def test_fresh_crossing_retains_both_linear_branches():
    grid = ParameterGrid.regular_grid(np.linspace(-1, 1, 7), np.linspace(-.7, .7, 7))
    x, y = grid.coordinates.T
    branches = np.column_stack([1+.25*(x+.13)+.03*y+.02j, 1-.25*(x+.13)+.03*y+.02j])
    values = tuple(v[::-1].copy() if s % 2 else v.copy() for s, v in enumerate(branches))
    data = BandCandidates(values)
    result = group_bands(grid, data)
    check_result(grid, data, result)
    for b in range(2):
        z = np.array([values[s][result.assignment[s, b]] for s in range(grid.n_sites)])
        assert any(np.all(z == branches[:, j]) for j in range(2))
    assert len(result.triangles) == 144


def test_import_has_only_numerical_runtime_dependencies():
    code = '''import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  # NumPy may import standard-library pickle internally; direct grouping I/O
  # imports are separately prohibited by the static source-boundary test.
  if fullname.split('.')[0] in {'myplots_app','tests','matplotlib','pandas','plotly','campaign_data','plot_workflows'}:
   raise ModuleNotFoundError('forbidden runtime dependency: '+fullname)
sys.meta_path.insert(0,Block())
import numpy as np
from eigenmode_analysis.band_grouping import ParameterGrid,BandCandidates,group_bands
c=ParameterGrid.regular_grid([-1.,0.,1.],[-1.,0.,1.])
f=BandCandidates(tuple(np.array([1.,2.]) for _ in range(9)))
r=group_bands(c,f)
assert r.n_candidates==18
print('NUMERIC_ONLY_OK')
'''
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'NUMERIC_ONLY_OK' in done.stdout


def test_no_consumer_or_io_imports_in_grouping_source():
    import eigenmode_analysis.band_grouping as module
    forbidden = {'myplots_app','tests','matplotlib','pandas','plotly','campaign_data','plot_workflows','pickle'}
    for path in Path(module.__file__).parent.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and not node.level else [])
            assert not any(n and n.split('.')[0] in forbidden for n in names), path
