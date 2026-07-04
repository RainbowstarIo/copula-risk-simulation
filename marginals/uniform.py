import numpy as np
from scipy.stats import uniform

def pdf(x : np.ndarray | float, a : float = 0.0, b : float = 1.0) -> np.ndarray:
    if b <= a:
        raise ValueError("b must be greater than a.")
    
    return uniform.pdf(x, loc = a, scale = b - a)

def cdf(x : np.ndarray | float, a : float = 0.0, b : float = 1.0) -> np.ndarray:
    if b <= a:
        raise ValueError("b must be greater than a.")
    
    return uniform.cdf(x, loc = a, scale = b - a)

def ppf(u : np.ndarray | float, a : float = 0.0, b : float = 1.0) -> np.ndarray:
    if b <= a:
        raise ValueError("b must be greater than a.")
    
    u = np.asarray(u, dtype = float)

    if np.any((u <= 0) | (u >= 1)) :
        raise ValueError("All input values must be in (0, 1).")
    
    return uniform.ppf(u, loc = a, scale = b - a)

def sample(n: int, a: float = 0.0, b: float = 1.0, seed: int | None = None) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    
    if b <= a:
        raise ValueError("b must be greater than a.")
    
    rng = np.random.default_rng(seed)

    return rng.uniform(low = a, high = b, size = n)
    