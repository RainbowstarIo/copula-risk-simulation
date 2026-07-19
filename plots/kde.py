import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_empirical_2d_density(
    samples: np.ndarray,
    title: str | None = None,
    xlabel: str = "X1",
    ylabel: str = "X2",
    grid_size: int = 100,
    levels: int = 15,
    show_points: bool = True
):
    """
    Plot an empirical two-dimensional density contour
    estimated from simulated samples using Gaussian KDE.
    """

    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 2:
        raise ValueError("samples must be a 2-dimensional array.")

    if samples.shape[1] != 2:
        raise ValueError(
            f"samples must have exactly 2 columns, got {samples.shape[1]}."
        )

    if samples.shape[0] < 2:
        raise ValueError("At least two sample points are required.")

    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1.")

    if levels <= 0:
        raise ValueError("levels must be positive.")

    x = samples[:, 0]
    y = samples[:, 1]

    x_padding = 0.05 * max(np.ptp(x), 1e-8)
    y_padding = 0.05 * max(np.ptp(y), 1e-8)

    x_grid = np.linspace(
        x.min() - x_padding,
        x.max() + x_padding,
        grid_size
    )

    y_grid = np.linspace(
        y.min() - y_padding,
        y.max() + y_padding,
        grid_size
    )

    X, Y = np.meshgrid(x_grid, y_grid)

    positions = np.vstack([
        X.ravel(),
        Y.ravel()
    ])

    values = np.vstack([
        x,
        y
    ])

    kde = gaussian_kde(values)

    Z = kde(positions).reshape(X.shape)

    fig, ax = plt.subplots()

    if show_points:
        ax.scatter(
            x,
            y,
            alpha=0.25,
            s=8
        )

    contour = ax.contour(
        X,
        Y,
        Z,
        levels=levels
    )
    """
    ax.clabel(
        contour,
        inline=True,
        fontsize=8
    )
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.grid(True)

    return fig, ax