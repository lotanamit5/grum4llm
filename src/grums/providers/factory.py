"""Construct preference providers without importing experiment runners."""

from __future__ import annotations

from typing import Any, Literal

from grums.contracts import PreferenceProvider
from grums.providers.huggingface import HuggingFaceProvider
from grums.providers.llm_stub import StubLLMPreferenceProvider
from grums.providers.oracle import OracleRankingProvider


def build_preference_provider(
    kind: Literal["oracle", "llm_stub", "huggingface"],
    *,
    ranking_by_agent_id: dict[str, tuple[int, ...]] | None = None,
    model: Any = None,
    tokenizer: Any = None,
    prompts_by_agent_id: dict[str, str] | None = None,
    alternative_texts: dict[int, str] | None = None,
) -> PreferenceProvider:
    if kind == "oracle":
        if ranking_by_agent_id is None:
            raise ValueError("oracle provider requires ranking_by_agent_id")
        return OracleRankingProvider(ranking_by_agent_id)
    if kind == "llm_stub":
        return StubLLMPreferenceProvider()
    if kind == "huggingface":
        if model is None or tokenizer is None or prompts_by_agent_id is None:
            raise ValueError("huggingface provider requires model, tokenizer, and prompts_by_agent_id")
        return HuggingFaceProvider(
            model=model, 
            tokenizer=tokenizer, 
            prompts_by_agent_id=prompts_by_agent_id,
            alternative_texts=alternative_texts,
        )
    raise ValueError(f"unknown provider kind: {kind!r}")
