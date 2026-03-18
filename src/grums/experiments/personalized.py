"""Personalized-choice experiment helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grums.core.parameters import GRUMParameters
from grums.experiments.metrics import personalized_mean_kendall_tau, raw_mean_kendall_tau
from grums.experiments.synthetic_data import make_dataset_1, make_dataset_2, make_dataset_consistency
from grums.inference import MCEMConfig, MCEMInference


@dataclass(frozen=True)
class PersonalizedPoint:
    n_agents: int
    mean_tau: float
    raw_mean_tau: float = 0.0


def _default_initial_params(m: int, k: int, l: int) -> GRUMParameters:
    return GRUMParameters(delta=np.zeros(m, dtype=float), interaction=np.zeros((k, l), dtype=float))


def run_personalized_asymptotic(
    agent_counts: list[int],
    repeats: int = 3,
    seed: int = 0,
    dataset: str = "dataset2",
    mcem_config: MCEMConfig | None = None,
) -> list[PersonalizedPoint]:
    """Personalized-choice analogue of asymptotic social-choice runner."""

    config = mcem_config or MCEMConfig(n_iterations=8, n_gibbs_samples=30, n_gibbs_burnin=15)
    points: list[PersonalizedPoint] = []

    for n in agent_counts:
        taus: list[float] = []
        raw_taus: list[float] = []
        for r in range(repeats):
            if dataset == "consistency":
                data = make_dataset_consistency(n_agents=max(agent_counts), seed=seed + r)
            elif dataset == "dataset1":
                data = make_dataset_1(n_agents=max(agent_counts), seed=seed + r)
            else:
                data = make_dataset_2(n_agents=max(agent_counts), seed=seed + r)
                
            init = _default_initial_params(
                m=data.params_true.n_alternatives,
                k=data.agent_features.shape[1],
                l=data.alternative_features.shape[1],
            )
            inf = MCEMInference(config)
            fit = inf.fit_map(
                initial_params=init,
                rankings=list(data.rankings[:n]),
                agent_features=data.agent_features[:n],
                alternative_features=data.alternative_features,
            )

            tau = personalized_mean_kendall_tau(
                params_true=data.params_true,
                params_est=fit.params,
                agent_features=data.agent_features,
                alternative_features=data.alternative_features,
            )
            taus.append(tau)

            raw_tau = raw_mean_kendall_tau(
                params_true=data.params_true,
                agent_features=data.agent_features,
                alternative_features=data.alternative_features,
                observed_rankings=list(data.rankings[:n]),
            )
            raw_taus.append(raw_tau)

        points.append(PersonalizedPoint(n_agents=n, mean_tau=float(np.mean(taus)), raw_mean_tau=float(np.mean(raw_taus))))

    return points
