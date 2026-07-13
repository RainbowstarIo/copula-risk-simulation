import numpy as np
import matplotlib.pyplot as plt

from plots.marginal import plot_2d_samples_with_marginals


def main():
    rng = np.random.default_rng(42)

    x = rng.normal(loc=0.0, scale=1.0, size=1000)
    y = 0.7 * x + rng.normal(loc=0.0, scale=0.8, size=1000)

    samples = np.column_stack([x, y])

    fig, ax = plot_2d_samples_with_marginals(
        samples,
        title="Test: Joint Samples with Marginal Histograms",
        xlabel="X1",
        ylabel="X2",
        bins=30,
        alpha=0.4,
        s=8
    )

    plt.show()


if __name__ == "__main__":
    main()