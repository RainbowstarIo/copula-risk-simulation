import numpy as np

def empirical_cdf(samples : np.ndarray, x : np.ndarray) -> float :

    samples = np.asarray(samples, dtype = float)
    x = np.asarray(x, dtype = float)

    if samples.ndim != 2:
        raise ValueError("samples must be a 2-dimensional array.")
    
    if x.ndim != 1:
        raise ValueError("x must be a 1-dimensional array.")
    
    if samples.shape[1] != x.shape[0]:
        raise ValueError(
            f"x must have length {samples.shape[1]}, got {x.shape[0]}."
        )
    
    indicators = np.all(samples <= x, axis=1)

    return float(np.mean(indicators))