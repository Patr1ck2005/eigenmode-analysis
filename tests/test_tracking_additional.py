from __future__ import annotations

import numpy as np
import pytest

from eigenmode_analysis.tracking import (
    infer_candidate_selector,
    pick_candidate_data,
    pick_candidate_data_stable,
)


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        (["a", "b"], ()),
        ([["a", "b"], ["c", "d"]], ("*",)),
        ([[["a", "b"], ["c", "d"]]], ("*", "*")),
    ],
)
def test_infer_candidate_selector_paths(cell, expected):
    assert infer_candidate_selector(cell, 2) == expected


def test_infer_candidate_selector_resolution_and_invalid_mode():
    cell = [["a", "b"], ["c", "d"]]
    assert infer_candidate_selector(cell, 2, resolve="deepest") == ("*",)
    assert infer_candidate_selector(cell, 2, resolve="shallowest") == ()
    with pytest.raises(ValueError, match="ambiguous"):
        infer_candidate_selector([[1, 2], [3, 4]], 2, resolve="error")
    with pytest.raises(ValueError, match="resolve"):
        infer_candidate_selector(cell, 2, resolve="other")


def test_selector_preserves_outer_structure_and_handles_missing_values():
    cell = np.array(
        [[{"mode": "a"}, {"mode": "b"}], [{"mode": "c"}, None]],
        dtype=object,
    )
    assert pick_candidate_data(cell, 1, ("*",)) == [{"mode": "b"}, None]
    assert pick_candidate_data(cell, 4, ("*",)) == [None, None]
    assert pick_candidate_data_stable(cell, 0, None) is None


def test_nested_selector_maps_each_outer_element():
    cell = [
        [["a", "b"], ["c", "d"]],
        [["e", "f"], ["g", "h"]],
    ]
    assert pick_candidate_data(cell, 1, ("*", "*")) == [["b", "d"], ["f", "h"]]
