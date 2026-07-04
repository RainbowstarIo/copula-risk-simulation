import numpy as np


def sample(n : int, dim : int = 2, seed : int | None = None) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    
    if dim <= 0:
        raise ValueError("dim must be positive.")
    #创建机器，还没有生成随机数
    rng = np.random.default_rng(seed)
    
    #这里是正式跑了
    z = rng.uniform(0.0, 1.0, size = (n, 1))

    #axis = 1沿着列方向，然后复制到dim
    return np.repeat(z, dim, axis = 1)

def cdf(u : np.ndarray) -> np.ndarray:

    u = np.asarray(u, dtype = float)

    if np.any((u < 0) | (u > 1)) :
        raise ValueError("All input values must be in [0, 1].")
    
    return np.min(u, axis = -1)


