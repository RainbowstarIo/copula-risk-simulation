import numpy as np
import pytest

from copulas import independence

def test_sample_shape():
    samples = independence.sample(n = 100, dim = 3, seed = 42)

    assert samples.shape == (100, 3)

def test_sample_values_are_in_unit_interval():
    samples = independence.sample(n = 100, dim = 3, seed = 42)

    assert np.all(samples >= 0)
    assert np.all(samples <= 1)

def test_cdf_known_value():
    u = np.array([0.4, 0.8])

    result = independence.cdf(u)
#0.4 * 0.8 copula
    assert np.isclose(result, 0.32)

def test_pdf_is_one():
    u = np.array([0.4, 0.8])

    result = independence.pdf(u)

    assert np.isclose(result, 1.0)

def test_invalid_sample_size_raises_error():
    with pytest.raises(ValueError):
        independence.sample(n = 0, dim = 2)


def test_invalid_dimension_raises_error():
    with pytest.raises(ValueError):
        independence.sample(n = 10, dim = 0)

def test_invalid_cdf_input_raises_error():
    with pytest.raises(ValueError):
        independence.cdf(np.array([0.5, 1.2]))

def test_vectorized_cdf():
    u = np.array([
        [0.4, 0.8],
        [0.5, 0.5],
        [1.0, 0.2],
    ])

    result = independence.cdf(u)

    expected = np.array([0.32, 0.25, 0.2])

    assert np.allclose(result, expected)