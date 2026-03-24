"""Synthetic datasets aligned with the paper's social-choice experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from grums.core.parameters import GRUMParameters

Tensor = torch.Tensor


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
    agent_features: Tensor
    alternative_features: Tensor
    rankings: tuple[tuple[int, ...], ...]
    sigma_noise: float


def _generate_dataset(config: SyntheticDatasetConfig, seed: int) -> SyntheticDataset:
    torch.manual_seed(seed)
    device = torch.device("cpu") # Generate on CPU for consistency, move to device later if needed

    x = torch.randn((config.n_agents, config.n_agent_features), dtype=torch.float64, device=device)
    z = torch.randn((config.n_alternatives, config.n_alternative_features), dtype=torch.float64, device=device)

    # Use Normal(1.0, 1.0) for delta as in the original code
    delta = config.delta_scale * (torch.randn(config.n_alternatives, dtype=torch.float64, device=device) + 1.0)
    interaction = torch.randn((config.n_agent_features, config.n_alternative_features), dtype=torch.float64, device=device)

    params_true = GRUMParameters(delta=delta, interaction=interaction)

    mu = x @ interaction @ z.T + delta.view(1, config.n_alternatives)
    noisy_u = mu + torch.randn(mu.shape, dtype=torch.float64, device=device) * config.sigma_noise
    
    # argsort along axis 1 (alternatives), descending
    rankings_tensor = torch.argsort(noisy_u, dim=1, descending=True)
    rankings = tuple(tuple(row.tolist()) for row in rankings_tensor)

    return SyntheticDataset(
        params_true=params_true,
        agent_features=x,
        alternative_features=z,
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


def make_dataset_consistency(
    n_agents: int = 100,
    n_alternatives: int = 5,
    n_agent_features: int = 2,
    n_alternative_features: int = 2,
    seed: int = 0,
) -> SyntheticDataset:
    """Generic dataset for consistency tests (Fig 2 and Fig 6) with sigma=1 and delta_scale=1."""

    config = SyntheticDatasetConfig(
        n_agents=n_agents,
        n_alternatives=n_alternatives,
        n_agent_features=n_agent_features,
        n_alternative_features=n_alternative_features,
        sigma_noise=1.0,
        delta_scale=1.0,
    )
    return _generate_dataset(config, seed)
