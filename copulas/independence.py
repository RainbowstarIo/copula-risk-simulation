import numpy as np
#import matplotlib.pyplot as plt

def sample(n : int, dim : int = 2, seed : int | None = None) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    
    if dim <= 0:
        raise ValueError("dim must be positive.")
    
    rng = np.random.default_rng(seed)

    return rng.uniform(0.0, 1.0, size = (n, dim))

def cdf(u : np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype = float)

    if np.any((u < 0) | (u > 1)):
        raise ValueError("All input values must be in [0, 1].")
    
    return np.prod(u, axis = -1)

def pdf(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)

    if np.any((u < 0) | (u > 1)):
        raise ValueError("All input values must be in [0, 1].")

    return np.ones(u.shape[:-1])

