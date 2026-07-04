import numpy as np
import pytest

from marginals import student_t


def test_student_t_pdf_basic():
    result = student_t.pdf(0.0, df=5)
    assert np.isfinite(result)
    assert result > 0.0


def test_student_t_cdf_basic():
    result = student_t.cdf(0.0, df=5)
    assert np.isclose(result, 0.5)


def test_student_t_ppf_basic():
    result = student_t.ppf(0.5, df=5)
    assert np.isclose(result, 0.0)


def test_student_t_ppf_shape():
    u = np.array([0.1, 0.5, 0.9])
    x = student_t.ppf(u, df=5)

    assert x.shape == u.shape
    assert np.all(np.isfinite(x))


def test_student_t_invalid_df():
    with pytest.raises(ValueError):
        student_t.ppf(0.5, df=0)


def test_student_t_ppf_invalid_u():
    with pytest.raises(ValueError):
        student_t.ppf(np.array([0.0, 0.5, 1.0]), df=5)