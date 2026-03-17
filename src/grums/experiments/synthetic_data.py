"""Synthetic datasets aligned with the paper's social-choice experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from grums.core.parameters import GRUMParameters

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    n_agents: int = 100
    n_alternatives: int = 5
    n_agent_features: int = 2
    n_alternative_features: int = 2
    sigma_noise: float = 1.0
    delta_scale: float = 1.0


@dataclass(frozen=True)
class SyntheticDataset:
    params_true: GRUMParameters
    agent_features: FloatArray
    alternative_features: FloatArray
    rankings: tuple[tuple[int, ...], ...]
    sigma_noise: float


def _generate_dataset(config: SyntheticDatasetConfig, seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(seed)

    x = rng.normal(0.0, 1.0, size=(config.n_agents, config.n_agent_features))
    z = rng.normal(0.0, 1.0, size=(config.n_alternatives, config.n_alternative_features))

    delta = config.delta_scale * rng.normal(1.0, 1.0, size=(config.n_alternatives,))
    interaction = rng.normal(0.0, 1.0, size=(config.n_agent_features, config.n_alternative_features))

    params_true = GRUMParameters(delta=delta.astype(float), interaction=interaction.astype(float))

    mu = x @ interaction @ z.T + delta.reshape(1, config.n_alternatives)
    noisy_u = mu + rng.normal(0.0, config.sigma_noise, size=mu.shape)
    rankings = tuple(tuple(np.argsort(-row)) for row in noisy_u)

    return SyntheticDataset(
        params_true=params_true,
        agent_features=x.astype(float),
        alternative_features=z.astype(float),
        rankings=rankings,
        sigma_noise=config.sigma_noise,
    )


def make_dataset_1(
    n_agents: int = 100,
    n_alternatives: int = 5,
    n_agent_features: int = 2,
    n_alternative_features: int = 2,
    seed: int = 0,
) -> SyntheticDataset:
    """Paper-style Dataset 1: weak social component and sigma=1."""

    config = SyntheticDatasetConfig(
        n_agents=n_agents,
        n_alternatives=n_alternatives,
        n_agent_features=n_agent_features,
        n_alternative_features=n_alternative_features,
        sigma_noise=1.0,
        delta_scale=0.1,
    )
    return _generate_dataset(config, seed)


def make_dataset_2(
    n_agents: int = 100,
    n_alternatives: int = 5,
    n_agent_features: int = 2,
    n_alternative_features: int = 2,
    seed: int = 0,
) -> SyntheticDataset:
    """Paper-style Dataset 2: stronger social component and lower noise."""

    config = SyntheticDatasetConfig(
        n_agents=n_agents,
        n_alternatives=n_alternatives,
        n_agent_features=n_agent_features,
        n_alternative_features=n_alternative_features,
        sigma_noise=0.5,
        delta_scale=1.0,
    )
    return _generate_dataset(config, seed)
