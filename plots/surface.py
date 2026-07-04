import numpy as np
import matplotlib.pyplot as plt


def plot_2d_function_surface(
    func,
    title: str | None = None,
    xlabel: str = "u",
    ylabel: str = "v",
    zlabel: str = "value",
    grid_size: int = 80,
    xlim: tuple[float, float] = (0.001, 0.999),
    ylim: tuple[float, float] = (0.001, 0.999),
    **func_params
):
    """
    Plot a 3D surface of a 2D function.

    This is useful for plotting a copula CDF or density.
    """

    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1.")

    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)

    X, Y = np.meshgrid(x, y)

    points = np.column_stack([
        X.ravel(),
        Y.ravel(),
    ])

    Z = func(points, **func_params)
    Z = np.asarray(Z, dtype=float).reshape(grid_size, grid_size)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(X, Y, Z)

    fig.colorbar(surface, ax=ax, shrink=0.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    if title is not None:
        ax.set_title(title)

    return fig, ax