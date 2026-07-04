import numpy as np
from scipy.stats import expon

def pdf(x : np.ndarray | float, lam : float = 1.0) -> np.ndarray:
    if lam <= 0:
        raise ValueError("lam must be positive.")
    
    return expon.pdf(x, loc = 0.0, scale = 1 / lam)

def cdf(x : np.ndarray | float, lam : float = 1.0) -> np.ndarray:
    if lam <= 0:
        raise ValueError("lam must be positive.")
    
    return expon.cdf(x, loc = 0.0, scale = 1 / lam)

def ppf(u: np.ndarray | float, lam: float = 1.0) -> np.ndarray:
    if lam <= 0:
        raise ValueError("lam must be positive.")
    
    u = np.asarray(u, dtype=float)

    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("All input values must be in (0, 1).")
    
    return expon.ppf(u, loc = 0.0, scale = 1 / lam)


def sample(n: int, lam: float = 1.0, seed: int | None = None) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    
    if lam <= 0:
        raise ValueError("lam must be positive.")
    
    rng = np.random.default_rng(seed)

    return rng.exponential(scale = 1 / lam, size = n)