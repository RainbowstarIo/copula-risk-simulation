import numpy as np
import pytest

from marginals import exponential

def test_exponential_pdf_basic():
    result = exponential.pdf(0.0, lam = 2.0)
    expected = 2.0

    assert np.isclose(result, expected)

def test_exponential_cdf_basic():
    result = exponential.cdf(0.0, lam = 2.0)

    assert np.isclose(result, 0.0)

def test_exponential_ppf_basic():
    result = exponential.ppf(0.5, lam = 2.0)
    expected = -np.log(1.0 - 0.5) / 2.0

    assert np.isclose(result, expected)

def test_exponential_ppf_shape():
    u = np.array([0.1, 0.5, 0.9])
    x = exponential.ppf(u, lam = 2.0)

    assert x.shape == u.shape
    assert np.all(np.isfinite(x))
    assert np.all(x >= 0.0)

def test_exponential_invali_lam():
    with pytest.raises(ValueError):
        exponential.ppf(0.5, lam = 0.0)

def test_exponential_ppf_invali_u():
    with pytest.raises(ValueError):
        exponential.ppf(np.array([0.0, 0.5, 1.0]), lam = 2.0)