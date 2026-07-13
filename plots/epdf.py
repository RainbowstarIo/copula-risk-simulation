import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_empirical_and_theoretical_pdf(
    samples: np.ndarray,
    theoretical_pdf,
    pdf_params: dict | None = None,
    title: str | None = None,
    xlabel: str = "x",
    bins: int = 30,
    grid_size: int = 500,
):
    """
    Plot a histogram, an empirical KDE, and a theoretical PDF
    for a one-dimensional sample.

    Parameters
    ----------
    samples:
        One-dimensional simulated sample.

    theoretical_pdf:
        The theoretical probability density function.

    pdf_params:
        Parameters passed to the theoretical PDF.

    title:
        Optional plot title.

    xlabel:
        Label of the x-axis.

    bins:
        Number of histogram bins.

    grid_size:
        Number of points used for plotting the density curves.

    Returns
    -------
    fig, ax:
        Matplotlib figure and axes.
    """

    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 1:
        raise ValueError("samples must be a 1-dimensional array.")

    if samples.size < 2:
        raise ValueError("samples must contain at least two values.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")

    if bins <= 0:
        raise ValueError("bins must be positive.")

    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1.")

    if pdf_params is None:
        pdf_params = {}

    sample_min = float(np.min(samples))
    sample_max = float(np.max(samples))

    # Avoid a zero-width plotting interval.
    if np.isclose(sample_min, sample_max):
        padding = 1.0
    else:
        padding = 0.05 * (sample_max - sample_min)

    x_grid = np.linspace(
        sample_min - padding,
        sample_max + padding,
        grid_size
    )

    # Kernel density estimate based on the simulated sample.
    kde = gaussian_kde(samples)
    kde_values = kde(x_grid)

    # Theoretical density based on the selected marginal distribution.
    theoretical_values = theoretical_pdf(
        x_grid,
        **pdf_params
    )

    fig, ax = plt.subplots()

    ax.hist(
        samples,
        bins=bins,
        density=True,
        alpha=0.4,
        label="Histogram"
    )

    ax.plot(
        x_grid,
        kde_values,
        label="Empirical KDE"
    )

    ax.plot(
        x_grid,
        theoretical_values,
        label="Theoretical PDF"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()

    if title is not None:
        ax.set_title(title)

    return fig, ax