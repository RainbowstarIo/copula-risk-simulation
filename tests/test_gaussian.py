import numpy as np
import pytest

from copulas import gaussian


def test_equicorrelation_matrix():
    P = gaussian.equicorrelation_matrix(dim = 3, rho = 0.5)

    expected = np.array([
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0],
    ])

    assert np.allclose(P, expected)

def test_sample_shape():
    P = gaussian.equicorrelation_matrix(dim = 3, rho = 0.5)

    samples = gaussian.sample(n = 100, P = P, seed = 42)

    assert samples.shape == (100, 3)

def test_sample_values_are_in_unit_interval():
    P = gaussian.equicorrelation_matrix(dim = 3, rho = 0.5)

    samples = gaussian.sample(n = 100, P = P, seed = 42)

    assert np.all(samples >= 0)
    assert np.all(samples <= 1)

def test_cdf_independence_case():
    P = np.eye(2)
    u = np.array([0.4, 0.8])

    result = gaussian.cdf(u, P)

    assert np.isclose(result, 0.32)

def test_pdf_independence_case():
    P = np.eye(2)
    u = np.array([0.4, 0.8])

    result = gaussian.pdf(u, P)

    assert np.isclose(result, 1.0)

def test_positive_correlation_increases_center_cdf():
    P = gaussian.equicorrelation_matrix(dim=2, rho=0.7)
    u = np.array([0.5, 0.5])

    result = gaussian.cdf(u, P)

    assert result > 0.25


def test_invalid_sample_size_raises_error():
    P = np.eye(2)

    with pytest.raises(ValueError):
        gaussian.sample(n=0, P=P)


def test_non_square_matrix_raises_error():
    P = np.array([
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
    ])

    with pytest.raises(ValueError):
        gaussian.sample(n=100, P=P)


def test_non_symmetric_matrix_raises_error():
    P = np.array([
        [1.0, 0.7],
        [0.2, 1.0],
    ])

    with pytest.raises(ValueError):
        gaussian.sample(n=100, P=P)


def test_invalid_cdf_input_raises_error():
    P = np.eye(2)

    with pytest.raises(ValueError):
        gaussian.cdf(np.array([0.5, 1.2]), P)


def test_invalid_pdf_boundary_input_raises_error():
    P = np.eye(2)

    with pytest.raises(ValueError):
        gaussian.pdf(np.array([0.0, 0.5]), P)   
