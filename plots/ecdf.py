import numpy as np
import matplotlib.pyplot as plt

def plot_empirical_and_theoretical_cdf(
        samples : np.ndarray,
        theoretical_cdf,
        cdf_params : dict | None = None,
        title : str | None = None,
        xlabel : str = "x",
):
    """
    Plot the empirical CDF of a one-dimensional sample together
    with the corresponding theoretical CDF.

    Parameters
    ----------
    samples:
        One-dimensional simulated sample.

    theoretical_cdf:
        The theoretical cumulative distribution function.

    cdf_params:
        Parameters passed to the theoretical CDF.

    title:
        Optional title of the plot.

    xlabel:
        Label of the x-axis.

    Returns
    -------
    fig, ax:
        Matplotlib figure and axes.
    """

    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 1:
        raise ValueError("samples must be a 1-dimensional array.")

    if samples.size == 0:
        raise ValueError("samples must not be empty.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")

    if cdf_params is None:
        cdf_params = {}

    # Sort the simulated observations.
    x_sorted = np.sort(samples)

    n = x_sorted.size

    # Empirical CDF values:
    # 1/n, 2/n, ..., n/n
    empirical_values = np.arange(1, n + 1) / n

    # Use a regular grid for the smooth theoretical CDF.
    x_grid = np.linspace(
        x_sorted[0],
        x_sorted[-1],
        500
    )

    theoretical_values = theoretical_cdf(
        x_grid,
        **cdf_params
    )

    fig, ax = plt.subplots()

    ax.step(
        x_sorted,
        empirical_values,
        where="post",
        label="Empirical CDF"
    )

    ax.plot(
        x_grid,
        theoretical_values,
        label="Theoretical CDF"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("F(x)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True)
    ax.legend()

    if title is not None:
        ax.set_title(title)

    return fig, ax