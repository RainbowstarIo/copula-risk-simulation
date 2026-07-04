import matplotlib.pyplot as plt

from copulas import gaussian
from plots.surface import plot_2d_function_surface


def main():
    P = gaussian.equicorrelation_matrix(dim=2, rho=0.7)

    plot_2d_function_surface(
    gaussian.pdf,
    title="Gaussian Copula Density Surface rho=0.7",
    xlabel="u",
    ylabel="v",
    zlabel="c(u, v)",
    grid_size=60,
    xlim=(0.02, 0.98),
    ylim=(0.02, 0.98),
    P=P,
)

    plt.show()


if __name__ == "__main__":
    main()