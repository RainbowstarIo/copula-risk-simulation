import numpy as np
import pytest

from simulation.joint_marginals import joint_sample


def dummy_copula_sample(n: int, dim: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.01, 0.99, size=(n, dim))


def test_joint_sample_shape():
    marginal_specs = [
        {"ppf": lambda u: u},
        {"ppf": lambda u: u},
    ]

    X = joint_sample(
        copula_sample_func=dummy_copula_sample,
        copula_params={"dim": 2},
        marginal_specs=marginal_specs,
        n=100,
        seed=42,
    )

    assert X.shape == (100, 2)
    assert np.all((X > 0) & (X < 1))


def test_joint_sample_with_params():
    def linear_ppf(u, a, b):
        return a * u + b

    marginal_specs = [
        {"ppf": linear_ppf, "params": {"a": 2.0, "b": 1.0}},
        {"ppf": linear_ppf, "params": {"a": 3.0, "b": -1.0}},
    ]

    X = joint_sample(
        copula_sample_func=dummy_copula_sample,
        copula_params={"dim": 2},
        marginal_specs=marginal_specs,
        n=50,
        seed=1,
    )

    assert X.shape == (50, 2)
    assert np.all(np.isfinite(X))


def test_joint_sample_invalid_n():
    marginal_specs = [
        {"ppf": lambda u: u},
    ]

    with pytest.raises(ValueError):
        joint_sample(
            copula_sample_func=dummy_copula_sample,
            copula_params={"dim": 1},
            marginal_specs=marginal_specs,
            n=0,
            seed=42,
        )