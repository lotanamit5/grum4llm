"""Ranking representations for GRUM preference data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FullRanking:
    """A complete ranking over alternatives 0..m-1."""

    order: tuple[int, ...]
    n_alternatives: int

    def __post_init__(self) -> None:
        if len(self.order) != self.n_alternatives:
            raise ValueError("full ranking length must equal n_alternatives")
        if len(set(self.order)) != len(self.order):
            raise ValueError("full ranking cannot contain duplicates")
        if set(self.order) != set(range(self.n_alternatives)):
            raise ValueError("full ranking must contain all alternatives exactly once")


@dataclass(frozen=True)
class PartialRanking:
    """A partial ranking over a subset of alternatives."""

    ordered_subset: tuple[int, ...]
    n_alternatives: int

    def __post_init__(self) -> None:
        if len(self.ordered_subset) == 0:
            raise ValueError("partial ranking must contain at least one alternative")
        if len(set(self.ordered_subset)) != len(self.ordered_subset):
            raise ValueError("partial ranking cannot contain duplicates")
        if min(self.ordered_subset) < 0 or max(self.ordered_subset) >= self.n_alternatives:
            raise ValueError("partial ranking alternative ids out of range")
