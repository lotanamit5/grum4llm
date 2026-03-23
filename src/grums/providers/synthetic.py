"""Synthetic data provider mapping to generic datasets ds0, ds1, ds2."""

from __future__ import annotations

import numpy as np

from grums.contracts import AgentRecord, AlternativeRecord, PreferenceProvider
from grums.experiments.synthetic_data import SyntheticDatasetConfig, _generate_dataset
from grums.providers.oracle import OracleRankingProvider


class SyntheticProvider(OracleRankingProvider):
    """An oracle ranking provider that wraps memory-generated synthetic rankings.
    
    Provides access to the underlying `dataset_record`, `agents`, `alternatives`,
    and `true_params` necessary for standard experiment evaluations.
    """

    def __init__(
        self,
        ds_name: str | None = None,
        n_agents: int = 100,
        n_alternatives: int = 5,
        n_agent_features: int = 2,
        n_alternative_features: int = 2,
        sigma_noise: float | None = None,
        delta_scale: float | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the synthetic data parameters and underlying rankings mapping."""
        if ds_name == "ds1":
            sigma_noise = sigma_noise if sigma_noise is not None else 1.0
            delta_scale = delta_scale if delta_scale is not None else 0.1
        elif ds_name == "ds2":
            sigma_noise = sigma_noise if sigma_noise is not None else 0.5
            delta_scale = delta_scale if delta_scale is not None else 1.0
        elif ds_name == "ds0":
            sigma_noise = sigma_noise if sigma_noise is not None else 1.0
            delta_scale = delta_scale if delta_scale is not None else 1.0
        else:
            sigma_noise = sigma_noise if sigma_noise is not None else 1.0
            delta_scale = delta_scale if delta_scale is not None else 1.0

        config = SyntheticDatasetConfig(
            n_agents=n_agents,
            n_alternatives=n_alternatives,
            n_agent_features=n_agent_features,
            n_alternative_features=n_alternative_features,
            sigma_noise=sigma_noise,
            delta_scale=delta_scale,
        )

        self.dataset_record = _generate_dataset(config, seed)
        self.true_params = self.dataset_record.params_true

        # Construct objects for elicitation engine interfaces
        self.agents = [
            AgentRecord(agent_id=f"agent_{i}", features=row)
            for i, row in enumerate(self.dataset_record.agent_features)
        ]
        self.alternatives = [
            AlternativeRecord(alternative_id=i, features=row)
            for i, row in enumerate(self.dataset_record.alternative_features)
        ]

        rankings_by_agent = {}
        for i, ranking in enumerate(self.dataset_record.rankings):
            rankings_by_agent[f"agent_{i}"] = ranking

        super().__init__(rankings_by_agent)
