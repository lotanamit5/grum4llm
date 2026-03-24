"""Synthetic data provider mapping to generic datasets ds0, ds1, ds2."""

from __future__ import annotations

import warnings
import torch

from grums.contracts import AgentRecord, AlternativeRecord
from grums.experiments.synthetic_data import SyntheticDatasetConfig, _generate_dataset
from grums.providers.oracle import OracleRankingProvider

DEFAULTS = {
    'ds0': {
        "n_agents": 500,
        "n_alternatives": 5,
        "n_agent_features": 2,
        "n_alternative_features": 2,
        "sigma_noise": 1.0,
        "delta_scale": 1.0
    },
    'ds1': {
        "n_agents": 100,
        "n_alternatives": 5,
        "n_agent_features": 2,
        "n_alternative_features": 2,
        "sigma_noise": 1.0,
        "delta_scale": 0.1
    }, 
    'ds2': {
        "n_agents": 100,
        "n_alternatives": 5,
        "n_agent_features": 2,
        "n_alternative_features": 2,
        "sigma_noise": 0.5,
        "delta_scale": 1.0
    }
}

class SyntheticProvider(OracleRankingProvider):
    """An oracle ranking provider that wraps memory-generated synthetic rankings.
    
    Provides access to the underlying `dataset_record`, `agents`, `alternatives`,
    and `true_params` necessary for standard experiment evaluations.
    """

    def __init__(
        self,
        ds_name: str | None = None,
        n_agents: int | None = None,
        n_alternatives: int | None = None,
        n_agent_features: int | None = None,
        n_alternative_features: int | None = None,
        sigma_noise: float | None = None,
        delta_scale: float | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the synthetic data parameters and underlying rankings mapping.
        
        Args:
            ds_name: Alias for standard synthetic benchmarks (ds0, ds1, ds2).
                     If provided, overrides structural kwargs.
            n_agents: n (Number of ranking agents in the entire population).
            n_alternatives: m (Number of candidates/alternatives).
            n_agent_features: K (Number of agent embedding features).
            n_alternative_features: L (Number of alternative embedding features).
            sigma_noise: Variance scalar for the noise term epsilon.
            delta_scale: Multiplier for the true population social choice array.
            seed: Random seed controlling dataset generations unconditionally.
        """
        provided_kwargs = {
            "n_agents": n_agents,
            "n_alternatives": n_alternatives,
            "n_agent_features": n_agent_features,
            "n_alternative_features": n_alternative_features,
            "sigma_noise": sigma_noise,
            "delta_scale": delta_scale,
        }
        
        if ds_name is not None and ds_name in DEFAULTS:
            custom_sent = any(v is not None for v in provided_kwargs.values())
            if custom_sent:
                warnings.warn(f"Warning: ds_name '{ds_name}' provided. Ignoring custom dataset parameters passed.")
            
            config_params = DEFAULTS[ds_name]
        else:
            fallbacks = {
                "n_agents": 100,
                "n_alternatives": 5,
                "n_agent_features": 2,
                "n_alternative_features": 2,
                "sigma_noise": 1.0,
                "delta_scale": 1.0
            }
            config_params = {k: (v if v is not None else fallbacks[k]) for k, v in provided_kwargs.items()}

        config = SyntheticDatasetConfig(**config_params)

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
