import numpy as np
from scipy.stats import t


def pdf(x: np.ndarray | float, df: float = 5.0, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    if df <= 0:
        raise ValueError("df must be positive.")
    
    if scale <= 0:
        raise ValueError("scale must be positive.")
    
    return t.pdf(x, df = df, loc = loc, scale = scale)


def cdf(x: np.ndarray | float, df: float = 5.0, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    if df <= 0:
        raise ValueError("df must be positive.")
    
    if scale <= 0:
        raise ValueError("scale must be positive.")
    
    return t.cdf(x, df = df, loc = loc, scale = scale)


def ppf(u: np.ndarray | float, df: float = 5.0, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    if df <= 0:
        raise ValueError("df must be positive.")
    
    if scale <= 0:
        raise ValueError("scale must be positive.")
    
    u = np.asarray(u, dtype=float)

    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("All input values must be in (0, 1).")
    
    return t.ppf(u, df = df, loc = loc, scale = scale)


def sample(
    n: int,
    df: float = 5.0,
    loc: float = 0.0,
    scale: float = 1.0,
    seed: int | None = None
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    
    if df <= 0:
        raise ValueError("df must be positive.")
    
    if scale <= 0:
        raise ValueError("scale must be positive.")
    
    rng = np.random.default_rng(seed)

    return loc + scale * rng.standard_t(df = df, size = n)