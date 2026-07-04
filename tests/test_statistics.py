import numpy as np
import pytest

from simulation.statistics import empirical_cdf


def test_empirical_cdf_basic():
    samples = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
    ])

    x = np.array([2.5, 3.5])

    result = empirical_cdf(samples, x)

    assert np.isclose(result, 0.5)


def test_empirical_cdf_all_true():
    samples = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ])

    x = np.array([10.0, 10.0])

    result = empirical_cdf(samples, x)

    assert np.isclose(result, 1.0)


def test_empirical_cdf_all_false():
    samples = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ])

    x = np.array([0.0, 0.0])

    result = empirical_cdf(samples, x)

    assert np.isclose(result, 0.0)


def test_empirical_cdf_invalid_samples_dimension():
    samples = np.array([1.0, 2.0, 3.0])
    x = np.array([2.0])

    with pytest.raises(ValueError):
        empirical_cdf(samples, x)


def test_empirical_cdf_invalid_x_dimension():
    samples = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ])

    x = np.array([[2.0, 3.0]])

    with pytest.raises(ValueError):
        empirical_cdf(samples, x)


def test_empirical_cdf_wrong_x_length():
    samples = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ])

    x = np.array([2.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        empirical_cdf(samples, x)