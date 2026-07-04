import numpy as np
from scipy.stats import t, multivariate_t


def _check_correlation_matrix(P: np.ndarray) -> np.ndarray:
    """
    Check whether P is a valid non-degenerate correlation matrix.

    Requirements:
    1. P is a square matrix.
    2. P is symmetric.
    3. diagonal entries are all 1.
    4. entries are in [-1, 1].
    5. P is positive definite.
    """

    P = np.asarray(P, dtype=float)

    if P.ndim != 2:
        raise ValueError("P must be a 2-dimensional matrix.")

    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix.")

    if not np.allclose(P, P.T):
        raise ValueError("P must be symmetric.")

    if not np.allclose(np.diag(P), 1.0):
        raise ValueError("P must have ones on the diagonal.")

    if np.any((P < -1) | (P > 1)):
        raise ValueError("All correlations in P must be in [-1, 1].")

    eigenvalues = np.linalg.eigvalsh(P)

    if np.any(eigenvalues <= 0):
        raise ValueError(
            "P must be positive definite. "
            "Degenerate boundary cases should be handled separately."
        )

    return P


def _check_df(df: float) -> float:
    """
    Check degrees of freedom for the t copula.
    """

    if df <= 0:
        raise ValueError("df must be positive.")

    return float(df)


def equicorrelation_matrix(dim: int, rho: float) -> np.ndarray:
    """
    Convenience function for the special case where all pairwise correlations
    are equal to rho.

    P_ij = rho for i != j
    P_ii = 1
    """

    if dim <= 1:
        raise ValueError("dim must be at least 2.")

    lower_bound = -1 / (dim - 1)

    if not (lower_bound < rho < 1):
        raise ValueError(
            f"For dim = {dim}, rho must be in ({lower_bound}, 1) "
            "for a non-degenerate equicorrelation matrix."
        )

    P = np.full((dim, dim), rho, dtype=float)
    np.fill_diagonal(P, 1.0)

    return P

def sample(
    n: int,
    P: np.ndarray,
    df: float,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate samples from a t copula with correlation matrix P
    and degrees of freedom df.

    Steps:
    1. Generate Z ~ N_d(0, P).
    2. Generate S ~ chi-square(df), independent of Z.
    3. Set X = Z / sqrt(S / df), so X has a multivariate t distribution.
    4. Transform each component by U_i = t_df(X_i).
    """

    if n <= 0:
        raise ValueError("n must be positive.")

    P = _check_correlation_matrix(P)
    df = _check_df(df)

    dim = P.shape[0]

    rng = np.random.default_rng(seed)

    z = rng.multivariate_normal(
        mean=np.zeros(dim),
        cov=P,
        size=n
    )

    s = rng.chisquare(df=df, size=n)

    x = z / np.sqrt(s[:, None] / df)

    u = t.cdf(x, df=df)

    return u

def sample_equicorrelation(
    n: int,
    dim: int = 2,
    rho: float = 0.7,
    df: float = 4.0,
    seed: int | None = None
) -> np.ndarray:
    P = equicorrelation_matrix(dim, rho)
    return sample(n=n, P=P, df=df, seed=seed)

def cdf(u: np.ndarray, P: np.ndarray, df: float) -> np.ndarray:
    """
    Evaluate the t copula CDF:

    C_{df, P}(u_1, ..., u_d)
    =
    T_{df, P}(t_df^{-1}(u_1), ..., t_df^{-1}(u_d)).
    """

    P = _check_correlation_matrix(P)
    df = _check_df(df)

    dim = P.shape[0]

    u = np.asarray(u, dtype=float)

    if np.any((u < 0) | (u > 1)):
        raise ValueError("All input values must be in [0, 1].")

    if u.shape[-1] != dim:
        raise ValueError(f"The last dimension of u must be {dim}.")

    x = t.ppf(u, df=df)

    return multivariate_t.cdf(
        x,
        loc=np.zeros(dim),
        shape=P,
        df=df
    )


def pdf(u: np.ndarray, P: np.ndarray, df: float) -> np.ndarray:
    """
    Evaluate the t copula density.

    Formula:

    c_{df, P}(u)
    =
    f_{df, P}(x_1, ..., x_d)
    /
    product_i f_df(x_i),

    where x_i = t_df^{-1}(u_i).
    """

    P = _check_correlation_matrix(P)
    df = _check_df(df)

    dim = P.shape[0]

    u = np.asarray(u, dtype=float)

    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("For pdf, all input values must be in (0, 1).")

    if u.shape[-1] != dim:
        raise ValueError(f"The last dimension of u must be {dim}.")

    x = t.ppf(u, df=df)

    numerator = multivariate_t.pdf(
        x,
        loc=np.zeros(dim),
        shape=P,
        df=df
    )

    denominator = np.prod(t.pdf(x, df=df), axis=-1)

    return numerator / denominator