import numpy as np

from grums.core import (
    interaction_design_matrix,
    is_interaction_identifiable,
    satisfies_connectivity_condition,
)


def test_connectivity_condition_true_for_connected_data() -> None:
    rankings = [
        (0, 1, 2),
        (1, 2, 0),
    ]
    assert satisfies_connectivity_condition(rankings, n_alternatives=3)


def test_connectivity_condition_false_for_disconnected_data() -> None:
    rankings = [
        (0, 1),
        (1, 0),
    ]
    assert not satisfies_connectivity_condition(rankings, n_alternatives=3)


def test_identifiability_rank_check() -> None:
    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    z = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    design = interaction_design_matrix(x, z)
    assert design.shape == (4, 4)
    assert is_interaction_identifiable(x, z)


def test_identifiability_detects_rank_deficiency() -> None:
    x = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float)
    z = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float)

    assert not is_interaction_identifiable(x, z)
