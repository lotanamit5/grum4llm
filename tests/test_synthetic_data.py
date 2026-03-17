import numpy as np

from grums.experiments.synthetic_data import make_dataset_1, make_dataset_2


def test_dataset_1_shapes() -> None:
    ds = make_dataset_1(n_agents=10, n_alternatives=4, n_agent_features=2, n_alternative_features=3, seed=1)

    assert ds.agent_features.shape == (10, 2)
    assert ds.alternative_features.shape == (4, 3)
    assert ds.params_true.delta.shape == (4,)
    assert ds.params_true.interaction.shape == (2, 3)
    assert len(ds.rankings) == 10


def test_dataset_2_has_lower_noise_than_dataset_1() -> None:
    ds1 = make_dataset_1(seed=2)
    ds2 = make_dataset_2(seed=2)

    assert ds2.sigma_noise < ds1.sigma_noise
    assert np.mean(np.abs(ds2.params_true.delta)) > np.mean(np.abs(ds1.params_true.delta))
