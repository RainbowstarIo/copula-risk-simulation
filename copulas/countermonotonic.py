import numpy as np

def sample(n : int , dim : int = 2, seed : int | None = None) -> np.ndarray:

    if n <= 0:
        raise ValueError("n must be positive.")
    
    if dim != 2:
        raise ValueError("Countermonotonoic copula is only implemented for dim = 2.")
    
    rng = np.random.default_rng(seed)

    u = rng.uniform(0.0, 1.0, size = n)
    v = 1.0 - u

    return np.column_stack([u, v])


def cdf(u : np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype = float)

    if u.shape[-1] != 2:
        raise ValueError("Countermonotonoic copula is only implemented for dim = 2.")
    
    if np.any((u < 0) | (u > 1)):
        raise ValueError("All input values must be in [0, 1].")
    
    return np.maximum(np.sum(u, axis = -1) - 1.0, 0.0)


    