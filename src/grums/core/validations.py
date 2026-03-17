"""Pre-inference checks for GRUM assumptions and identifiability."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def satisfies_connectivity_condition(
    rankings: list[tuple[int, ...]],
    n_alternatives: int,
) -> bool:
    """Check Condition 1 from the paper.

    For every partition C1, C2 of alternatives, at least one pair c1 in C1, c2 in C2
    must appear in some ranking with c1 preferred to c2.
    """

    all_alts = set(range(n_alternatives))
    if n_alternatives < 2:
        return False

    for r in range(1, n_alternatives):
        for subset in combinations(range(n_alternatives), r):
            c1 = set(subset)
            c2 = all_alts - c1
            if not c2:
                continue

            has_cross_edge = False
            for ranking in rankings:
                rank_pos = {alt: idx for idx, alt in enumerate(ranking)}
                for left in c1:
                    for right in c2:
                        left_pos = rank_pos.get(left)
                        right_pos = rank_pos.get(right)
                        if left_pos is not None and right_pos is not None and left_pos < right_pos:
                            has_cross_edge = True
                            break
                    if has_cross_edge:
                        break
                if has_cross_edge:
                    break

            if not has_cross_edge:
                return False

    return True


def interaction_design_matrix(
    agent_features: FloatArray,
    alternative_features: FloatArray,
) -> FloatArray:
    """Build design rows for vec(B) with kron(z_j, x_i)."""

    if agent_features.ndim != 2 or alternative_features.ndim != 2:
        raise ValueError("feature matrices must be 2D")

    rows: list[np.ndarray] = []
    for x in agent_features:
        for z in alternative_features:
            rows.append(np.kron(z, x))
    return np.vstack(rows)


def is_interaction_identifiable(
    agent_features: FloatArray,
    alternative_features: FloatArray,
    tol: float = 1e-10,
) -> bool:
    """Check rank condition for B identifiability in linearized model."""

    design = interaction_design_matrix(agent_features, alternative_features)
    rank = np.linalg.matrix_rank(design, tol=tol)
    k = agent_features.shape[1]
    l = alternative_features.shape[1]
    return rank == (k * l)
