import pytest

from grums.core import FullRanking, PartialRanking


def test_full_ranking_valid() -> None:
    ranking = FullRanking(order=(2, 0, 1), n_alternatives=3)
    assert ranking.order[0] == 2


def test_full_ranking_rejects_missing_item() -> None:
    with pytest.raises(ValueError, match="must contain all alternatives"):
        _ = FullRanking(order=(0, 1, 4), n_alternatives=3)


def test_partial_ranking_valid() -> None:
    ranking = PartialRanking(ordered_subset=(3, 1), n_alternatives=5)
    assert ranking.ordered_subset == (3, 1)


def test_partial_ranking_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _ = PartialRanking(ordered_subset=(0, 5), n_alternatives=5)
