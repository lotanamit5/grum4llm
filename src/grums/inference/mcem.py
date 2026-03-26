"""Monte Carlo EM inference for Normal-family GRUM with Pairwise DAG Constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from grums.contracts import Observation, compile_constraint_graph
from grums.core.model_math import compute_mean_utilities
from grums.core.parameters import GRUMParameters
from grums.core.validations import satisfies_connectivity_condition
import logging

Tensor = torch.Tensor


@dataclass(frozen=True)
class MCEMConfig:
    n_iterations: int = 20
    n_gibbs_samples: int = 100
    n_gibbs_burnin: int = 50
    sigma: float = 1.0
    prior_precision: float = 1e-2
    tolerance: float = 1e-5
    random_seed: int = 0
    connectivity_check_every: int = 1
    connectivity_start_at: int = 1


@dataclass(frozen=True)
class MCEMResult:
    params: GRUMParameters
    objective_trace: tuple[float, ...]
    converged: bool
    n_iterations: int


class MCEMInference:
    """Algorithm-3 style MC-EM for Normal utility noise utilizing PyTorch Graph constraints."""

    def __init__(self, config: MCEMConfig | None = None) -> None:
        self.config = config or MCEMConfig()
        if not math.isfinite(self.config.sigma) or self.config.sigma <= 0.0:
            raise ValueError("sigma must be a finite positive value")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compile_adjacencies(
        self,
        graph: dict[str, list[tuple[int, int]]],
        unique_agent_ids: list[str],
        n_alternatives: int,
    ) -> tuple[Tensor, Tensor]:
        """Compile pairwise directed graphs into boolean adjacency arrays matching parallel operations."""
        n_agents = len(unique_agent_ids)
        adj_upper = torch.zeros((n_agents, n_alternatives, n_alternatives), dtype=torch.bool, device=self.device)
        adj_lower = torch.zeros((n_agents, n_alternatives, n_alternatives), dtype=torch.bool, device=self.device)
        
        for i, agent_id in enumerate(unique_agent_ids):
            edges = graph.get(agent_id, [])
            for winner, loser in edges:
                adj_lower[i, winner, loser] = True # loser is a lower bound for winner
                adj_upper[i, loser, winner] = True # winner is an upper bound for loser
                
        return adj_upper, adj_lower

    def fit_map(
        self,
        initial_params: GRUMParameters,
        observations: list[Observation],
        agent_features: Tensor,
        alternative_features: Tensor,
        fit_bt: bool = False,
    ) -> MCEMResult:
        params = GRUMParameters(
            delta=initial_params.delta.to(self.device),
            interaction=initial_params.interaction.to(self.device)
        )
        if fit_bt:
            # Force B=0 for Bradley-Terry baseline
            params.interaction.zero_()
        
        alternative_features = alternative_features.to(self.device)
        
        # Consolidation logic: group observations by unique agent_id
        # This ensures that one agent = one latent vector in the E-step
        graph = compile_constraint_graph(observations)
        unique_agent_ids = list(graph.keys())
        n_unique_agents = len(unique_agent_ids)
        
        # Map agent_id to its first encountered features row for consistency
        agent_id_to_first_idx = {}
        for i, obs in enumerate(observations):
            if obs.agent_id not in agent_id_to_first_idx:
                agent_id_to_first_idx[obs.agent_id] = i
        
        consolidated_features = torch.stack([
            agent_features[agent_id_to_first_idx[aid]] for aid in unique_agent_ids
        ]).to(self.device)
        
        n_alts = int(params.n_alternatives)
        
        # Identifiability Check: Condition 1 (Strong Connectivity)
        n_obs = len(observations)
        if n_obs >= self.config.connectivity_start_at and (n_obs - self.config.connectivity_start_at) % self.config.connectivity_check_every == 0:
            def _get_ranking(o: Observation) -> tuple[int, ...]:
                if hasattr(o, "ranking"):
                    return o.ranking  # type: ignore
                return (o.winner_id, o.loser_id)  # type: ignore

            rankings = [_get_ranking(o) for o in observations]
            if not satisfies_connectivity_condition(rankings, n_alts):
                logging.warning(
                    f"Condition 1 (Strong Connectivity) not satisfied at step {n_obs}. "
                    "The MLE/MAP estimate may be unbounded."
                )
            
        adj_upper, adj_lower = self._compile_adjacencies(graph, unique_agent_ids, n_alts)
        
        objective_trace: list[float] = []
        torch.manual_seed(self.config.random_seed)
        
        converged = False
        for step in range(1, self.config.n_iterations + 1):
            s = self._e_step(params, adj_upper, adj_lower, consolidated_features, alternative_features)
            new_params = self._m_step(
                s, params, consolidated_features, alternative_features, fit_bt=fit_bt
            )

            q_val = self._q_objective(s, new_params, consolidated_features, alternative_features)
            objective_trace.append(float(q_val))

            param_diff = torch.linalg.norm(new_params.delta - params.delta) + torch.linalg.norm(
                new_params.interaction - params.interaction
            )
            params = new_params

            if param_diff < self.config.tolerance:
                converged = True
                return MCEMResult(
                    params=params,
                    objective_trace=tuple(objective_trace),
                    converged=converged,
                    n_iterations=step,
                )

        return MCEMResult(
            params=params,
            objective_trace=tuple(objective_trace),
            converged=False,
            n_iterations=self.config.n_iterations,
        )

    def _e_step(
        self,
        params: GRUMParameters,
        adj_upper: Tensor,
        adj_lower: Tensor,
        agent_features: Tensor,
        alternative_features: Tensor,
    ) -> Tensor:
        
        mu = compute_mean_utilities(params, agent_features, alternative_features)
        n_agents, m = mu.shape
        sigma = self.config.sigma
        
        current_U = mu.clone()
        sqrt2 = math.sqrt(2.0)
        
        collected_sums = torch.zeros_like(current_U)
        total = self.config.n_gibbs_burnin + self.config.n_gibbs_samples
        
        for t in range(total):
            for j in range(m):
                u_upper = current_U.clone()
                u_upper[~adj_upper[:, j, :]] = float('inf')
                upper_bound, _ = u_upper.min(dim=1)
                
                u_lower = current_U.clone()
                u_lower[~adj_lower[:, j, :]] = float('-inf')
                lower_bound, _ = u_lower.max(dim=1)
                
                # numerical safety evaluating divergent bounds
                invalid = lower_bound >= upper_bound
                lower_bound[invalid] = mu[invalid, j] - 10.0 * sigma
                upper_bound[invalid] = mu[invalid, j] + 10.0 * sigma
                
                alpha = (lower_bound - mu[:, j]) / sigma
                beta = (upper_bound - mu[:, j]) / sigma
                
                alpha = torch.nan_to_num(alpha, nan=-10.0, posinf=-10.0, neginf=-10.0)
                beta = torch.nan_to_num(beta, nan=10.0, posinf=10.0, neginf=10.0)
                
                p_a = 0.5 * (1.0 + torch.erf(alpha / sqrt2))
                p_b = 0.5 * (1.0 + torch.erf(beta / sqrt2))
                
                p = torch.rand(n_agents, device=self.device) * (p_b - p_a) + p_a
                
                # Safety clamp preventing NaNs
                p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
                
                sample = mu[:, j] + sigma * sqrt2 * torch.erfinv(2.0 * p - 1.0)
                
                sample = torch.max(sample, lower_bound + 1e-5)
                sample = torch.min(sample, upper_bound - 1e-5)
                
                current_U[:, j] = sample
                
            if t >= self.config.n_gibbs_burnin:
                collected_sums += current_U
                
        return collected_sums / self.config.n_gibbs_samples

    def _m_step(
        self,
        s_matrix: Tensor,
        prev_params: GRUMParameters,
        agent_features: Tensor,
        alternative_features: Tensor,
        fit_bt: bool = False,
    ) -> GRUMParameters:
        sigma2 = self.config.sigma**2
        lam = self.config.prior_precision

        n_agents, n_alts = s_matrix.shape
        k = agent_features.size(1)
        l = alternative_features.size(1)

        xbzt = agent_features @ prev_params.interaction @ alternative_features.T
        delta = (s_matrix - xbzt).sum(dim=0) / (n_agents + lam * sigma2)

        y = (s_matrix - delta.view(1, n_alts)).reshape(-1)
        
        # Kronecker products matrix evaluation natively
        # A bit memory hungry but incredibly fast over GPU
        
        rows = []
        for x in agent_features:
            for z in alternative_features:
                rows.append(torch.kron(x, z).unsqueeze(0))
        design = torch.cat(rows, dim=0)

        ridge = lam * sigma2 * torch.eye(k * l, device=self.device)
        lhs = design.T @ design + ridge
        rhs = design.T @ y
        
        b_vec = torch.linalg.solve(lhs, rhs)
        b_matrix = b_vec.view(k, l)
        if fit_bt:
            b_matrix.zero_()

        # Enforce Identifiability: Sum-to-zero constraint on delta
        # This prevents the global origin drift observed in unbounded runs
        delta = delta - delta.mean()

        return GRUMParameters(delta=delta, interaction=b_matrix)

    def _q_objective(
        self,
        s_matrix: Tensor,
        params: GRUMParameters,
        agent_features: Tensor,
        alternative_features: Tensor,
    ) -> float:
        sigma2 = self.config.sigma**2
        lam = self.config.prior_precision

        mu = compute_mean_utilities(params, agent_features, alternative_features)
        residual = s_matrix - mu
        data_term = -0.5 / sigma2 * float(torch.sum(residual**2))
        prior_term = -0.5 * lam * float(torch.sum(params.delta**2) + torch.sum(params.interaction**2))
        return data_term + prior_term
