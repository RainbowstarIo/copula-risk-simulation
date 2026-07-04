import numpy as np
import pytest

from copulas import countermonotonic

def test_sample_shape():
    samples = countermonotonic.sample(n = 100, dim = 2, seed = 42)

    assert samples.shape == (100, 2)

def test_sample_values_are_in_unit_interval():
    samples = countermonotonic.sample(n = 100, dim = 2, seed = 42)

    assert np.all(samples >= 0)
    assert np.all(samples <= 1)

def test_sample_columns_sum_to_one():
    samples = countermonotonic.sample(n = 100, dim = 2, seed = 42)

    assert np.allclose(samples[:, 0] + samples[:, 1], 1.0)

def test_cdf_known_value():
    u = np.array([0.7, 0.6])

    result = countermonotonic.cdf(u)

    assert np.isclose(result, 0.3)


def test_vectorized_cdf():
    u = np.array([
        [0.7, 0.6],
        [0.2, 0.3],
        [0.9, 0.9],
    ])

    result = countermonotonic.cdf(u)

    expected = np.array([0.3, 0.0, 0.8])

    assert np.allclose(result, expected)


def test_invalid_sample_size_raises_error():
    with pytest.raises(ValueError):
        countermonotonic.sample(n=0, dim=2)


def test_invalid_dimension_raises_error():
    with pytest.raises(ValueError):
        countermonotonic.sample(n=10, dim=3)


def test_invalid_cdf_dimension_raises_error():
    with pytest.raises(ValueError):
        countermonotonic.cdf(np.array([0.5, 0.5, 0.5]))


def test_invalid_cdf_input_raises_error():
    with pytest.raises(ValueError):
        countermonotonic.cdf(np.array([0.5, 1.2]))