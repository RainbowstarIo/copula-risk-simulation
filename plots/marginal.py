import numpy as np
import matplotlib.pyplot as plt


def plot_2d_samples_with_marginals(
        samples : np.ndarray,
        title : str | None = None,
        xlabel : str = "X1",
        ylabel : str = "X2",
        bins : int = 30,
        alpha : float = 0.4,
        s : float = 8.0
):
    """
    Plot two-dimensional samples together with marginal histograms.

    The main plot shows the joint sample.
    The upper histogram shows the marginal distribution of X1.
    The right histogram shows the marginal distribution of X2.
    """

    samples = np.asarray(samples, dtype = float)

    if samples.ndim != 2:
        raise ValueError("samples must be a 2-dimensional array.")
    
    if samples.shape[1] != 2:
        raise ValueError(
            f"samples must have exactly 2 columns, got {samples.shape[1]}."
        )
    
    if bins <= 0:
        raise ValueError("bins must be positive.")
    
    x = samples[:, 0]
    y = samples[:, 1]

    fig = plt.figure(figsize = (8, 7))

    grid = fig.add_gridspec(
        2,
        2,
        width_ratios = (4, 1.2),
        height_ratios = (1.2, 4),
        hspace = 0.1,
        wspace = 0.1
    )

    ax_hist_x = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[1, 0])
    ax_hist_y = fig.add_subplot(grid[1, 1])

    ax_scatter.scatter(
        x,
        y,
        alpha = alpha,
        s = s
    )

    ax_hist_x.hist(
        x,
        bins = bins,
        density = True
    )

    ax_hist_y.hist(
        y,
        bins = bins,
        density = True,
        orientation = "horizontal"
    )

    ax_hist_x.set_xlim(ax_scatter.get_xlim())
    ax_hist_y.set_ylim(ax_scatter.get_ylim())

    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)
    ax_scatter.grid(True)

  
    ax_hist_y.tick_params(
    axis="y",
    labelleft=True,
    labelright=False,
    pad=0
    )

    ax_hist_y.tick_params(
    axis="x",
    pad=1
    )

   

    ax_hist_x.set_ylabel("Density")
    ax_hist_y.set_xlabel("Density")

    if title is not None:
        fig.suptitle(title, y = 0.98)

    return fig, ax_scatter