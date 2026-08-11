"""Schema-neutral helpers for mapping per-candidate auxiliary data."""

from __future__ import annotations

from typing import Any

import numpy as np


def infer_candidate_selector(
    cell: Any,
    n_candidates: int,
    *,
    resolve: str = "deepest",
) -> tuple[str, ...] | None:
    """Infer the nested path to a candidate axis.

    ``()`` means that *cell* itself is the candidate sequence. Each ``"*"``
    denotes one sequence level to map through before indexing the candidate.
    """

    if n_candidates < 0:
        raise ValueError("n_candidates must be non-negative")
    if resolve not in {"deepest", "shallowest", "error"}:
        raise ValueError("resolve must be 'deepest', 'shallowest', or 'error'")

    def safe_len(value: Any) -> int | None:
        try:
            return len(value)
        except (TypeError, AttributeError):
            return None

    def is_sequence(value: Any) -> bool:
        return isinstance(value, (list, tuple, np.ndarray))

    paths: list[tuple[str, ...]] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        length = safe_len(node)
        if length is None:
            return
        if length == n_candidates:
            paths.append(path)
        if is_sequence(node) and length:
            visit(node[0], path + ("*",))

    visit(cell, ())
    unique = sorted(set(paths), key=len)
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    if resolve == "error":
        raise ValueError(
            "auxiliary cell has ambiguous candidate-axis depths: "
            f"{unique}; provide an explicit selector"
        )
    return unique[-1] if resolve == "deepest" else unique[0]


def pick_candidate_data(cell: Any, candidate_index: int, selector: tuple[str, ...] | None):
    """Pick one candidate's auxiliary data while preserving outer structure."""

    if cell is None or selector is None:
        return None
    if not isinstance(candidate_index, (int, np.integer)):
        raise TypeError("candidate_index must be an integer")

    def descend(node: Any, depth: int):
        if depth == len(selector):
            try:
                if not 0 <= candidate_index < len(node):
                    return None
                return node[candidate_index]
            except (TypeError, AttributeError, IndexError):
                return None
        if selector[depth] != "*":
            raise ValueError("selector tokens must be '*'")
        try:
            length = len(node)
        except (TypeError, AttributeError):
            return None
        return [descend(node[index], depth + 1) for index in range(length)]

    return descend(cell, 0)


def pick_candidate_data_stable(
    cell: Any,
    candidate_index: int,
    selector: tuple[str, ...] | None,
):
    """Stable wrapper that returns ``None`` when no selector is available."""

    return pick_candidate_data(cell, candidate_index, selector)


__all__ = [
    "infer_candidate_selector",
    "pick_candidate_data",
    "pick_candidate_data_stable",
]
