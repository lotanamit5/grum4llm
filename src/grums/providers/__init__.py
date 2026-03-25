"""Provider adapters for external preference sources.

Core elicitation code must depend only on ``PreferenceProvider`` (see ``grums.contracts``).
Implementations here are source-agnostic at the type level: observations carry no simulator/LLM label.
"""

from grums.providers.factory import build_preference_provider
from grums.providers.huggingface import HuggingFaceProvider, MockHuggingFaceProvider
from grums.providers.llm_stub import StubLLMPreferenceProvider
from grums.providers.oracle import OracleRankingProvider

__all__ = [
    "HuggingFaceProvider",
    "MockHuggingFaceProvider",
    "OracleRankingProvider",
    "StubLLMPreferenceProvider",
    "build_preference_provider",
]
