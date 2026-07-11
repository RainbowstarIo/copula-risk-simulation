import numpy as np
import matplotlib.pyplot as plt

def plot_2d_samples(
        samples : np.ndarray,
        title : str | None = None,
        xlabel : str = "X1",
        ylabel : str = "X2",
        alpha : float = 0.5,
        s : float = 15.0,
):
    samples = np.asarray(samples, dtype = float)

    if samples.ndim != 2:
        raise ValueError("samples must be a 2-dimensional array.")
    
    if samples.shape[1] != 2:
        raise ValueError(
            f"samples must have exactly 2 columns for a 2D scatter plot, got {samples.shape[1]}."
        )
    

    fig, ax = plt.subplots(figsize = (5, 4))

    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        alpha=alpha,
        s=s,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.grid(True)

    return fig, ax  