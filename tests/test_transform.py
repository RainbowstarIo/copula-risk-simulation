import numpy as np
import pytest

from simulation.transform import apply_marginals

def test_apply_marginals_identity_ppf():
    U = np.array([
        [0.2, 0.7],
        [0.5, 0.9],
        [0.8, 0.1],
    ])

    marginal_specs = [
        {"ppf" : lambda u : u},
        {"ppf" : lambda u : u},
    ]

    X = apply_marginals(U, marginal_specs)

    assert np.allclose(X, U)

def test_apply_marginals_with_params():
    U = np.array([
        [0.2, 0.7],
        [0.5, 0.9],
    ])

    def linear_ppf(u, a, b):
        return a * u + b

    marginal_specs = [
        {"ppf": linear_ppf, "params": {"a": 2.0, "b": 1.0}},
        {"ppf": linear_ppf, "params": {"a": 10.0, "b": 0.0}},
    ]

    X = apply_marginals(U, marginal_specs)

    expected = np.array([
        [1.4, 7.0],
        [2.0, 9.0],
    ])

    assert np.allclose(X, expected)


def test_apply_marginals_wrong_dimension():
    U = np.array([0.2, 0.5, 0.8])

    marginal_specs = [
        {"ppf": lambda u: u},
    ]

    with pytest.raises(ValueError):
        apply_marginals(U, marginal_specs)


def test_apply_marginals_wrong_number_of_specs():
    U = np.array([
        [0.2, 0.7],
        [0.5, 0.9],
    ])

    marginal_specs = [
        {"ppf": lambda u: u},
    ]

    with pytest.raises(ValueError):
        apply_marginals(U, marginal_specs)


def test_apply_marginals_invalid_u_values():
    U = np.array([
        [0.0, 0.7],
        [0.5, 1.0],
    ])

    marginal_specs = [
        {"ppf": lambda u: u},
        {"ppf": lambda u: u},
    ]

    with pytest.raises(ValueError):
        apply_marginals(U, marginal_specs)