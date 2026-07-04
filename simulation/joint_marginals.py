import numpy as np
from simulation.transform import apply_marginals


def joint_sample(
    copula_sample_func,
    copula_params: dict,
    marginal_specs: list[dict],
    n: int,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate joint samples using a copula and given marginal distributions.

    Parameters
    ----------
    copula_sample_func:
        Function that generates copula samples U in (0, 1)^d.

    copula_params:
        Parameters for the copula sample function.

    marginal_specs:
        List of marginal specifications.

    n:
        Number of samples.

    seed:
        Random seed.

    Returns
    -------
    X:
        Joint samples with specified marginals and copula dependence.
    """

    if n <= 0:
        raise ValueError("n must be positive.")

    U = copula_sample_func(n=n, seed=seed, **copula_params)

    X = apply_marginals(U, marginal_specs)

    return X