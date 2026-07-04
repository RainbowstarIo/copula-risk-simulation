import numpy as np
import pytest

from copulas import t_copula


def test_equicorrelation_matrix():
    P = t_copula.equicorrelation_matrix(dim=3, rho=0.5)

    expected = np.array([
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0],
    ])

    assert np.allclose(P, expected)


def test_sample_shape():
    P = t_copula.equicorrelation_matrix(dim=3, rho=0.5)

    samples = t_copula.sample(n=100, P=P, df=4, seed=42)

    assert samples.shape == (100, 3)


def test_sample_values_are_in_unit_interval():
    P = t_copula.equicorrelation_matrix(dim=3, rho=0.5)

    samples = t_copula.sample(n=100, P=P, df=4, seed=42)

    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_cdf_returns_probability():
    P = np.eye(2)
    u = np.array([0.4, 0.8])

    result = t_copula.cdf(u, P, df=4)

    assert 0 <= result <= 1


def test_pdf_is_positive():
    P = np.eye(2)
    u = np.array([0.4, 0.8])

    result = t_copula.pdf(u, P, df=4)

    assert result > 0


def test_large_df_pdf_close_to_independence():
    P = np.eye(2)
    u = np.array([0.4, 0.8])

    result = t_copula.pdf(u, P, df=1000)

    assert np.isclose(result, 1.0, atol=1e-2)


def test_invalid_sample_size_raises_error():
    P = np.eye(2)

    with pytest.raises(ValueError):
        t_copula.sample(n=0, P=P, df=4)


def test_invalid_df_raises_error():
    P = np.eye(2)

    with pytest.raises(ValueError):
        t_copula.sample(n=100, P=P, df=0)


def test_invalid_cdf_input_raises_error():
    P = np.eye(2)

    with pytest.raises(ValueError):
        t_copula.cdf(np.array([0.5, 1.2]), P, df=4)


def test_invalid_pdf_boundary_input_raises_error():
    P = np.eye(2)

    with pytest.raises(ValueError):
        t_copula.pdf(np.array([0.0, 0.5]), P, df=4)