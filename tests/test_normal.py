import numpy as np
import pytest

from marginals import normal

def test_normal_pdf_basic():
    result = normal.pdf(0.0, mu = 0.0, sigma = 1.0)
    expected = 1 / np.sqrt(2 * np.pi)

    assert np.isclose(result, expected)

def test_normal_cdf_basic():
    result = normal.cdf(0.0, mu = 0.0, sigma = 1.0)

    assert np.isclose(result, 0.5)

def test_normal_ppf_basic():
    result = normal.ppf(0.5, mu = 0.0, sigma = 1.0)

    assert np.isclose(result, 0.0)

def test_normal_ppf_shape():
    u = np.array([0.1, 0.5, 0.9])
    x = normal.ppf(u, mu = 0.0, sigma = 1.0)

    assert x.shape == u.shape
    assert np.all(np.isfinite(x))

def test_normal_invalid_sigma():
    with pytest.raises(ValueError):
        normal.ppf(0.5, mu = 0.0, sigma = 0.0)

def test_normal_ppf_invalid_u():
    with pytest.raises(ValueError):
        normal.ppf(np.array([0.0, 0.5, 1.0]), mu = 0.0, sigma = 1.0)