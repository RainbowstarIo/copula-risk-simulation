import numpy as np


def apply_marginals(U: np.ndarray, marginal_specs: list[dict]) -> np.ndarray:
    U = np.asarray(U, dtype=float)

    if U.ndim != 2:
        raise ValueError("U must be a 2-dimensional array.")

    if np.any((U <= 0) | (U >= 1)):
        raise ValueError("All values in U must be in (0, 1).")

    n, dim = U.shape

    if len(marginal_specs) != dim:
        raise ValueError(
            f"Expected {dim} marginal specifications, got {len(marginal_specs)}."
        )

    X = np.empty_like(U, dtype=float)

    for j, spec in enumerate(marginal_specs):
        ppf = spec["ppf"]
        params = spec.get("params", {})

        X[:, j] = ppf(U[:, j], **params)

    return X