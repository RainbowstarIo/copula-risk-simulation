import numpy as np
import pytest

from marginals import uniform


def test_uniform_pdf_basic():
    result = uniform.pdf(0.5, a=0.0, b=1.0)
    expected = 1.0

    assert np.isclose(result, expected)


def test_uniform_cdf_basic():
    result = uniform.cdf(0.5, a=0.0, b=1.0)

    assert np.isclose(result, 0.5)


def test_uniform_ppf_basic():
    result = uniform.ppf(0.5, a=0.0, b=1.0)

    assert np.isclose(result, 0.5)


def test_uniform_ppf_shape():
    u = np.array([0.1, 0.5, 0.9])
    x = uniform.ppf(u, a=2.0, b=4.0)

    assert x.shape == u.shape
    assert np.all(np.isfinite(x))
    assert np.all((x >= 2.0) & (x <= 4.0))


def test_uniform_invalid_bounds():
    with pytest.raises(ValueError):
        uniform.ppf(0.5, a=1.0, b=1.0)


def test_uniform_ppf_invalid_u():
    with pytest.raises(ValueError):
        uniform.ppf(np.array([0.0, 0.5, 1.0]), a=0.0, b=1.0)