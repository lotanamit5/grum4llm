"""Experiment helpers for reproducible GRUM evaluations."""

from grums.experiments.benchmark import (
    compare_criteria_social_choice,
    run_asymptotic_social_choice,
)
from grums.experiments.metrics import moving_average, personalized_mean_kendall_tau, social_choice_kendall_tau
from grums.experiments.personalized import PersonalizedPoint, run_personalized_asymptotic
from grums.experiments.synthetic_data import (
    SyntheticDataset,
    SyntheticDatasetConfig,
    make_dataset_1,
    make_dataset_2,
)

__all__ = [
    "SyntheticDataset",
    "SyntheticDatasetConfig",
    "make_dataset_1",
    "make_dataset_2",
    "social_choice_kendall_tau",
    "personalized_mean_kendall_tau",
    "moving_average",
    "run_asymptotic_social_choice",
    "compare_criteria_social_choice",
    "PersonalizedPoint",
    "run_personalized_asymptotic",
]
