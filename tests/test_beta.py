import numpy as np
import pytest

from marginals import beta


def test_beta_pdf_basic():
    result = beta.pdf(0.5, alpha=2.0, beta_param=2.0)

    assert np.isfinite(result)
    assert result > 0.0


def test_beta_cdf_basic():
    result = beta.cdf(0.5, alpha=2.0, beta_param=2.0)

    assert np.isclose(result, 0.5)


def test_beta_ppf_basic():
    result = beta.ppf(0.5, alpha=2.0, beta_param=2.0)

    assert np.isclose(result, 0.5)


def test_beta_ppf_shape():
    u = np.array([0.1, 0.5, 0.9])
    x = beta.ppf(u, alpha=2.0, beta_param=5.0)

    assert x.shape == u.shape
    assert np.all(np.isfinite(x))
    assert np.all((x >= 0.0) & (x <= 1.0))


def test_beta_invalid_alpha():
    with pytest.raises(ValueError):
        beta.ppf(0.5, alpha=0.0, beta_param=2.0)


def test_beta_invalid_beta_param():
    with pytest.raises(ValueError):
        beta.ppf(0.5, alpha=2.0, beta_param=0.0)


def test_beta_ppf_invalid_u():
    with pytest.raises(ValueError):
        beta.ppf(np.array([0.0, 0.5, 1.0]), alpha=2.0, beta_param=2.0)