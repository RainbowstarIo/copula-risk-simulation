import numpy as np
import matplotlib.pyplot as plt


def plot_2d_function_contour(
    func,
    title: str | None = None,
    xlabel: str = "u",
    ylabel: str = "v",
    grid_size: int = 100,
    xlim: tuple[float, float] = (0.001, 0.999),
    ylim: tuple[float, float] = (0.001, 0.999),
    levels: int = 20,
    **func_params
):
    """
    Plot contour lines of a 2D function.

    This is useful for plotting copula CDF or copula density.

    Parameters
    ----------
    func:
        Function to evaluate. It should accept points of shape (n, 2).

    title:
        Optional plot title.

    xlabel:
        Label for x-axis.

    ylabel:
        Label for y-axis.

    grid_size:
        Number of grid points per axis.

    xlim:
        Range of x-axis.

    ylim:
        Range of y-axis.

    levels:
        Number of contour levels.

    func_params:
        Extra parameters passed to func.

    Returns
    -------
    fig, ax:
        Matplotlib figure and axes.
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

    fig, ax = plt.subplots()

    contour = ax.contourf(X, Y, Z, levels=levels)
    fig.colorbar(contour, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    return fig, ax