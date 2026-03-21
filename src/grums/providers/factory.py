"""Construct preference providers without importing experiment runners."""

from __future__ import annotations

from typing import Literal

from grums.contracts import PreferenceProvider
from grums.providers.llm_stub import StubLLMPreferenceProvider
from grums.providers.oracle import OracleRankingProvider


def build_preference_provider(
    kind: Literal["oracle", "llm_stub"],
    *,
    ranking_by_agent_id: dict[str, tuple[int, ...]] | None = None,
) -> PreferenceProvider:
    if kind == "oracle":
        if ranking_by_agent_id is None:
            raise ValueError("oracle provider requires ranking_by_agent_id")
        return OracleRankingProvider(ranking_by_agent_id)
    if kind == "llm_stub":
        return StubLLMPreferenceProvider()
    raise ValueError(f"unknown provider kind: {kind!r}")
