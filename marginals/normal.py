import numpy as np
from scipy.stats import norm

def pdf(x : np.ndarray | float, mu : float = 0.0, sigma : float = 1.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    
    return norm.pdf(x, loc = mu, scale = sigma)

def cdf(x : np.ndarray | float, mu : float = 0.0, sigma : float = 1.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    
    return norm.cdf(x, loc = mu, scale = sigma)

def ppf(u : np.ndarray | float, mu : float = 0.0, sigma : float = 1.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    
    u = np.asarray(u, dtype = float)

    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("All input values must be in (0, 1).")
    
    return norm.ppf(u, loc = mu, scale = sigma)

def sample(n : int, mu : float = 0.0, sigma : float = 1.0, seed : int | None = None) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    
    rng = np.random.default_rng(seed)

    return rng.normal(loc = mu, scale = sigma, size = n)