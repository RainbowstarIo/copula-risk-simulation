import numpy as np
from scipy.stats import norm, multivariate_normal

def _check_correlation_matrix(P : np.ndarray) -> np.ndarray:
    """
    Check whether P is a valid non-degenerate correlation matrix.

    Requirements:
    1. P is a square matrix.
    2. P is symmetric.
    3. diagonal entries are all 1.
    4. entries are in [-1, 1].
    5. P is positive definite.
    """
    
    P = np.asarray(P, dtype = float)

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
            "Degenerate boundary cases such as rho = 1 should be handled "
            "by comonotonic copula, and rho = -1 in dimension 2 by "
            "countermonotonic copula."
        )
    
    return P

def equicorrelation_matrix(dim : int, rho : float) -> np.ndarray:
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
    
    P = np.full((dim, dim), rho, dtype = float)
    np.fill_diagonal(P, 1.0)

    return P


def sample(n : int, P : np.ndarray, seed : int | None = None) -> np.ndarray:
    """
    Generate samples from a Gaussian copula with correlation matrix P.

    Steps:
    1. Generate Z ~ N_d(0, P).
    2. Transform each component by U_i = Phi(Z_i).
    3. Return U, whose marginals are U(0, 1).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    
    P = _check_correlation_matrix(P)

    dim = P.shape[0]

    rng = np.random.default_rng(seed)

    z = rng.multivariate_normal(
        mean = np.zeros(dim),
        cov = P,
        size = n
    )

    u = norm.cdf(z)

    return u

def sample_equicorrelation(
    n: int,
    dim: int = 2,
    rho: float = 0.7,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate samples from a Gaussian copula with an equicorrelation matrix.

    This is a convenience wrapper for the case where all pairwise
    correlations are equal to rho.
    """
    P = equicorrelation_matrix(dim, rho)
    return sample(n=n, P=P, seed=seed)


def cdf(u : np.ndarray, P : np.ndarray) -> np.ndarray:
    """
    Evaluate the Gaussian copula CDF:

    C_P(u_1, ..., u_d)
    = Phi_P(Phi^{-1}(u_1), ..., Phi^{-1}(u_d)).
    """

    P = _check_correlation_matrix(P)

    dim = P.shape[0]

    u = np.asarray(u, dtype = float)

    if np.any((u < 0) | (u > 1)):
        raise ValueError("All input values must be in [0, 1].")
    
    if u.shape[-1] != dim :
        raise ValueError(f"The last dimension of u must be {dim}.")
    
    x = norm.ppf(u)

    return multivariate_normal.cdf(
        x,
        mean = np.zeros(dim),
        cov = P
    )

def pdf(u : np.ndarray, P : np.ndarray) -> np.ndarray:
    """
    Evaluate the Gaussian copula density.

    Formula:

    c_P(u)
    = phi_P(x_1, ..., x_d) / product_i phi(x_i),

    where x_i = Phi^{-1}(u_i).
    """

    P = _check_correlation_matrix(P)
    dim = P.shape[0]

    u = np.asarray(u, dtype = float)

    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("For pdf, all input values must be in (0, 1).")
    
    if u.shape[-1] != dim:
        raise ValueError(f"The last dimension of u must be {dim}.")
    
    x = norm.ppf(u)

    numerator = multivariate_normal.pdf(
        x,
        mean = np.zeros(dim),
        cov = P
    )

    denominator = np.prod(norm.pdf(x), axis = -1)

    return numerator / denominator

