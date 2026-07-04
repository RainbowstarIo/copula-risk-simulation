import numpy as np
from scipy.stats import beta


def pdf(x: np.ndarray | float, alpha: float = 2.0, beta_param: float = 2.0) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    
    if beta_param <= 0:
        raise ValueError("beta_param must be positive.")
    
    return beta.pdf(x, a = alpha, b = beta_param)


def cdf(x: np.ndarray | float, alpha: float = 2.0, beta_param: float = 2.0) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    
    if beta_param <= 0:
        raise ValueError("beta_param must be positive.")
    
    return beta.cdf(x, a = alpha, b = beta_param)


def ppf(u: np.ndarray | float, alpha: float = 2.0, beta_param: float = 2.0) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    
    if beta_param <= 0:
        raise ValueError("beta_param must be positive.")
    
    u = np.asarray(u, dtype = float)

    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("All input values must be in (0, 1).")
    
    return beta.ppf(u, a = alpha, b = beta_param)


def sample(
    n: int,
    alpha: float = 2.0,
    beta_param: float = 2.0,
    seed: int | None = None
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    
    if beta_param <= 0:
        raise ValueError("beta_param must be positive.")
    
    rng = np.random.default_rng(seed)

    return rng.beta(a=alpha, b=beta_param, size=n)