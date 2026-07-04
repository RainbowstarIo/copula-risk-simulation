import numpy as np
import pytest

from copulas import comonotonic

def test_sample_shape():
    samples = comonotonic.sample(n = 100, dim = 3, seed = 42)

    assert samples.shape == (100, 3)

def test_sample_values_are_in_unit_intervall():
    samples = comonotonic.sample(n = 100, dim = 3, seed = 42)

    assert np.all(samples >= 0)
    assert np.all(samples <= 1)

def test_sample_columns_are_equal():
    samples = comonotonic.sample(n = 100, dim = 3, seed = 42)
#samples[:, 0]取第一列
    assert np.allclose(samples[:, 0], samples[:, 1])
    assert np.allclose(samples[:, 0], samples[:, 2])

def test_cdf_known_value():
    u = np.array([0.3, 0.7, 0.5])

    result = comonotonic.cdf(u)

    assert np.isclose(result, 0.3)

def test_vectorized_cdf():
    u = np.array([
        [0.3, 0.7, 0.5],
        [0.8, 0.2, 0.6],
        [0.9, 0.9, 0.9],
    ])

    result = comonotonic.cdf(u)

    expected = np.array([0.3, 0.2, 0.9])

    assert np.allclose(result, expected)

def test_invalid_sample_size_raises_error():
    with pytest.raises(ValueError):
        comonotonic.sample(n=0, dim=2)


def test_invalid_dimension_raises_error():
    with pytest.raises(ValueError):
        comonotonic.sample(n=10, dim=0)


def test_invalid_cdf_input_raises_error():
    with pytest.raises(ValueError):
        comonotonic.cdf(np.array([0.5, 1.2]))   