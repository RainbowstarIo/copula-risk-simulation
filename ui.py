import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copulas import gaussian
from copulas import t_copula
from copulas import independence
from copulas import comonotonic
from copulas import countermonotonic

from marginals import normal
from marginals import exponential
from marginals import student_t
from marginals import uniform
from marginals import beta

from simulation.transform import apply_marginals

from plots.scatter import plot_2d_samples
from plots.contour import plot_2d_function_contour
from plots.surface import plot_2d_function_surface
from plots.marginal import plot_2d_samples_with_marginals
from plots.ecdf import plot_empirical_and_theoretical_cdf
from plots.epdf import plot_empirical_and_theoretical_pdf
from plots.kde import plot_empirical_2d_density


st.set_page_config(
    page_title="Copula Risk Simulation",
    layout="wide"
)

st.title("Copula Risk Simulation")

st.write(
    """
    Generate dependent bivariate samples by combining a copula with two marginal distributions.

    The copula controls the dependence structure, while the marginal distributions control
    the behavior of the individual random variables.
    """
)


def build_copula(copula_name: str, rho: float | None = None, df: float | None = None):
    """
    Build copula sample function and CDF function.

    Returns
    -------
    copula_sample_func:
        Function for generating U samples.

    copula_sample_params:
        Parameters for the sample function.

    copula_cdf_func:
        Function for evaluating the copula CDF.

    copula_cdf_params:
        Parameters for the CDF function.
    """

   
    if copula_name == "Gaussian":
        if rho is None :
            raise ValueError("rho is required for the Gaussian copula.")
        
        P = np.array([
            [1.0, rho],
            [rho, 1.0]
        ])
        return (
            gaussian.sample,
            {"P": P},
            gaussian.cdf,
            {"P": P},
            gaussian.pdf,
            {"P": P}
        )

    if copula_name == "t":
        if rho is None :
            raise ValueError("rho is required for the t copula.")
        
        if df is None :
            raise ValueError("df is required for the t copula.")
        
        P = np.array([
            [1.0, rho],
            [rho, 1.0]
        ])

        return (
            t_copula.sample,
            {"P": P, "df": df},
            t_copula.cdf,
            {"P": P, "df": df},
            t_copula.pdf,
            {"P": P, "df": df}
        )

    if copula_name == "Independence":
        return (
            independence.sample,
            {"dim": 2},
            independence.cdf,
            {},
            independence.pdf,
            {}
        )

    if copula_name == "Comonotonic":
        return (
            comonotonic.sample,
            {"dim": 2},
            comonotonic.cdf,
            {},
            None,
            {}
        )

    if copula_name == "Countermonotonic":
        return (
            countermonotonic.sample,
            {"dim": 2},
            countermonotonic.cdf,
            {},
            None,
            {}
        )

    raise ValueError(f"Unknown copula: {copula_name}")


def build_marginal_spec(marginal_name: str, prefix: str) -> dict:
    """
    Build one marginal distribution specification.

    The returned dictionary contains:
    - ppf function
    - parameters for the ppf
    """

    if marginal_name == "Normal":
        mu = st.sidebar.number_input(
            f"{prefix} Normal mu",
            value=0.0
        )

        sigma = st.sidebar.number_input(
            f"{prefix} Normal sigma",
            min_value=0.0001,
            value=1.0
        )

        return {
            "ppf": normal.ppf,
            "cdf": normal.cdf,
            "pdf": normal.pdf,
            "params": {"mu": mu, "sigma": sigma}
        }

    if marginal_name == "Exponential":
        lam = st.sidebar.number_input(
            f"{prefix} Exponential lambda",
            min_value=0.0001,
            value=1.0
        )

        return {
            "ppf": exponential.ppf,
            "cdf": exponential.cdf,
            "pdf": exponential.pdf,
            "params": {"lam": lam}
        }

    if marginal_name == "Student-t":
        df = st.sidebar.number_input(
            f"{prefix} Student-t df",
            min_value=0.0001,
            value=5.0
        )

        loc = st.sidebar.number_input(
            f"{prefix} Student-t loc",
            value=0.0
        )

        scale = st.sidebar.number_input(
            f"{prefix} Student-t scale",
            min_value=0.0001,
            value=1.0
        )

        return {
            "ppf": student_t.ppf,
            "cdf": student_t.cdf,
            "pdf": student_t.pdf,
            "params": {"df": df, "loc": loc, "scale": scale}
        }

    if marginal_name == "Uniform":
        a = st.sidebar.number_input(
            f"{prefix} Uniform a",
            value=0.0
        )

        b = st.sidebar.number_input(
            f"{prefix} Uniform b",
            value=1.0
        )

        return {
            "ppf": uniform.ppf,
            "cdf": uniform.cdf,
            "pdf": uniform.pdf,
            "params": {"a": a, "b": b}
        }

    if marginal_name == "Beta":
        alpha = st.sidebar.number_input(
            f"{prefix} Beta alpha",
            min_value=0.0001,
            value=2.0
        )

        beta_param = st.sidebar.number_input(
            f"{prefix} Beta beta",
            min_value=0.0001,
            value=5.0
        )

        return {
            "ppf": beta.ppf,
            "cdf": beta.cdf,
            "pdf": beta.pdf,
            "params": {"alpha": alpha, "beta_param": beta_param}
        }

    raise ValueError(f"Unknown marginal distribution: {marginal_name}")


# ======================================================
# Sidebar
# ======================================================

st.sidebar.header("General Settings")

n = st.sidebar.number_input(
    "Number of samples",
    min_value=100,
    max_value=100000,
    value=1000,
    step=100
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    value=42,
    step=1
)

st.sidebar.header("Copula Settings")

copula_name = st.sidebar.selectbox(
    "Copula",
    [
        "Gaussian",
        "t",
        "Independence",
        "Comonotonic",
        "Countermonotonic"
    ]
)

rho = None
df_copula = None

if copula_name in ["Gaussian", "t"]:
        rho = st.sidebar.slider(
            "Correlation rho",
            min_value = -0.95,
            max_value = 0.95,
            value = 0.7,
            step = 0.05
        )

if copula_name == "t":
    df_copula = st.sidebar.number_input(
        "Degrees of freedom for t copula",
        min_value = 0.0001,
        value = 4.0
    )

st.sidebar.header("Marginal Distributions")

marginal_1 = st.sidebar.selectbox(
    "Marginal for X1",
    [
        "Normal",
        "Exponential",
        "Student-t",
        "Uniform",
        "Beta"
    ],
    key="marginal_1"
)

marginal_2 = st.sidebar.selectbox(
    "Marginal for X2",
    [
        "Normal",
        "Exponential",
        "Student-t",
        "Uniform",
        "Beta"
    ],
    key="marginal_2"
)

st.sidebar.subheader("Parameters for X1")

marginal_spec_1 = build_marginal_spec(
    marginal_name=marginal_1,
    prefix="X1"
)

st.sidebar.subheader("Parameters for X2")

marginal_spec_2 = build_marginal_spec(
    marginal_name=marginal_2,
    prefix="X2"
)

marginal_specs = [
    marginal_spec_1,
    marginal_spec_2
]

st.sidebar.header("Plot Settings")

show_u_scatter = st.sidebar.checkbox(
    "Show copula samples U",
    value=True
)

show_x_scatter = st.sidebar.checkbox(
    "Show transformed samples X",
    value=True
)

show_marginal_plot = st.sidebar.checkbox(
    "Show marginal histograms",
    value = True
)

show_u_empirical_density = st.sidebar.checkbox(
    "Show empirical 2D density of U",
    value = True
)

show_x_empirical_density = st.sidebar.checkbox(
    "Show empirical 2D density of X",
    value = True
)

show_empirical_cdf = st.sidebar.checkbox(
    "Show empirical marginal CDF plots.",
    value = True
)

show_empirical_pdf = st.sidebar.checkbox(
    "Show empirical marginal PDF plots.",
    value = True
)

show_contour = st.sidebar.checkbox(
    "Show copula CDF contour plot",
    value=True
)

show_surface = st.sidebar.checkbox(
    "Show copula CDF surface plot",
    value=True
)

show_pdf_contour = st.sidebar.checkbox(
    "Show copula PDF contour plot",
    value=True
)

show_pdf_surface = st.sidebar.checkbox(
    "Show copula PDF surface plot",
    value=True
)

grid_size = st.sidebar.slider(
    "Grid size for contour/surface",
    min_value=20,
    max_value=100,
    value=50,
    step=10
)

levels = st.sidebar.slider(
    "Contour levels",
    min_value=5,
    max_value=50,
    value=20,
    step=5
)


# ======================================================
# Main Button
# ======================================================

if st.button("Generate Samples"):
    try:
        st.write("Step 1: Building copula...")

        (
            copula_sample_func,
            copula_sample_params,
            copula_cdf_func,
            copula_cdf_params,
            copula_pdf_func,
            copula_pdf_params
        ) = build_copula(
            copula_name=copula_name,
            rho=float(rho) if rho is not None else None,
            df=float(df_copula) if df_copula is not None else None
        )

        st.write("Step 2: Generating copula samples U...")

        U = copula_sample_func(
            n=int(n),
            seed=int(seed),
            **copula_sample_params
        )

        st.write("Step 3: Applying marginal transformations...")

        X = apply_marginals(
            U=U,
            marginal_specs=marginal_specs
        )

        st.success("Samples generated successfully.")

        # ======================================================
        # DataFrames
        # ======================================================

        st.subheader("Generated Samples")

        df_u = pd.DataFrame(
            U,
            columns=["U1", "U2"]
        )

        df_x = pd.DataFrame(
            X,
            columns=["X1", "X2"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("Copula samples U")
            st.write(f"Shape of U: {U.shape}")
            st.dataframe(df_u.head(20))

        with col2:
            st.write("Transformed samples X")
            st.write(f"Shape of X: {X.shape}")
            st.dataframe(df_x.head(20))

        # ======================================================
        # Summary Statistics
        # ======================================================

        st.subheader("Summary Statistics")

        st.write("Summary statistics of copula samples U")
        st.dataframe(df_u.describe())

        st.write("Summary statistics of transformed samples X")
        st.dataframe(df_x.describe())

        st.write("Sample correlation matrix of X")
        st.dataframe(df_x.corr())

        st.write("Sample correlation matrix of U")
        st.dataframe(df_u.corr())

        # ======================================================
        # Scatter Plots
        # ======================================================

        if show_u_scatter:
            st.subheader("Scatter Plot of Copula Samples U")

            fig_u, ax_u = plot_2d_samples(
                U,
                title=f"{copula_name} copula samples in U-space",
                xlabel="U1",
                ylabel="U2",
                alpha=0.4,
                s=8
            )

            st.pyplot(fig_u)
            plt.close(fig_u)

        if show_x_scatter:
            st.subheader("Scatter Plot of Transformed Samples X")

            fig_x, ax_x = plot_2d_samples(
                X,
                title=f"{copula_name} copula with {marginal_1} and {marginal_2} marginals",
                xlabel="X1",
                ylabel="X2",
                alpha=0.4,
                s=8
            )

            st.pyplot(fig_x)
            plt.close(fig_x)

        if show_marginal_plot :
            st.subheader("Joint Samples with Marginal Histograms")

            fig_marginal, ax_marginal = plot_2d_samples_with_marginals(
                X,
                title = (
                    f"{copula_name} copula with "
                    f"{marginal_1} and {marginal_2} marginals"
                ),
                xlabel = "X1",
                ylabel = "X2",
                bins = 30,
                alpha = 0.55,
                s = 10
            )
        
            st.pyplot(fig_marginal)
            plt.close(fig_marginal)



        # ======================================================
        # Empirical 2D Density Plots
        # ======================================================

        if show_u_empirical_density:
            st.subheader("Empirical 2D Density of Copula Samples U")

            try:
                fig_u_density, ax_u_density = plot_empirical_2d_density(
                    U,
                    title=f"Empirical 2D density of {copula_name} copula samples",
                    xlabel="U1",
                    ylabel="U2",
                    grid_size=int(grid_size),
                    levels=int(levels),
                    show_points=True
                )

                st.pyplot(fig_u_density)
                plt.close(fig_u_density)

            except np.linalg.LinAlgError:
                st.info(
                    "Empirical 2D density is not available for this copula "
                    "because the samples lie on a lower-dimensional line."
                )

        if show_x_empirical_density:
            st.subheader("Empirical 2D Density of Transformed Samples X")

            try:
                fig_x_density, ax_x_density = plot_empirical_2d_density(
                    X,
                    title=(
                        f"Empirical 2D density of {copula_name} copula with "
                        f"{marginal_1} and {marginal_2} marginals"
                    ),
                    xlabel="X1",
                    ylabel="X2",
                    grid_size=int(grid_size),
                    levels=int(levels),
                    show_points=True
                )

                st.pyplot(fig_x_density)
                plt.close(fig_x_density)

            except np.linalg.LinAlgError:
                st.info(
                    "Empirical 2D density is not available for this sample "
                    "because the data lie on a lower-dimensional line."
                )

        

        # ======================================================
        # Empirical Marginal CDF Plots
        # ======================================================

        if show_empirical_cdf:
            st.subheader("Empirical and Theoretical Marginal CDFs")

            col_cdf_1, col_cdf_2 = st.columns(2)

            with col_cdf_1:
                fig_cdf_1, ax_cdf_1 = plot_empirical_and_theoretical_cdf(
                    samples=X[:, 0],
                    theoretical_cdf=marginal_spec_1["cdf"],
                    cdf_params=marginal_spec_1["params"],
                    title=f"Empirical vs Theoretical CDF of X1 ({marginal_1})",
                    xlabel="X1"
                )

                st.pyplot(fig_cdf_1)
                plt.close(fig_cdf_1)

            with col_cdf_2:
                fig_cdf_2, ax_cdf_2 = plot_empirical_and_theoretical_cdf(
                    samples=X[:, 1],
                    theoretical_cdf=marginal_spec_2["cdf"],
                    cdf_params=marginal_spec_2["params"],
                    title=f"Empirical vs Theoretical CDF of X2 ({marginal_2})",
                    xlabel="X2"
                )

                st.pyplot(fig_cdf_2)
                plt.close(fig_cdf_2)

        # ======================================================
        # Empirical Marginal PDF Plots
        # ======================================================

        if show_empirical_pdf:
            st.subheader("Empirical and Theoretical Marginal PDFs")

            col_pdf_1, col_pdf_2 = st.columns(2)

            with col_pdf_1:
                fig_epdf_1, ax_epdf_1 = (
                    plot_empirical_and_theoretical_pdf(
                        samples=X[:, 0],
                        theoretical_pdf=marginal_spec_1["pdf"],
                        pdf_params=marginal_spec_1["params"],
                        title=f"Empirical vs Theoretical PDF of X1 ({marginal_1})",
                        xlabel="X1",
                        bins=30
                    )
                )

                st.pyplot(fig_epdf_1)
                plt.close(fig_epdf_1)

            with col_pdf_2:
                fig_epdf_2, ax_epdf_2 = (
                    plot_empirical_and_theoretical_pdf(
                        samples=X[:, 1],
                        theoretical_pdf=marginal_spec_2["pdf"],
                        pdf_params=marginal_spec_2["params"],
                        title=f"Empirical vs Theoretical PDF of X2 ({marginal_2})",
                        xlabel="X2",
                        bins=30
                    )
                )

                st.pyplot(fig_epdf_2)
                plt.close(fig_epdf_2)



        # ======================================================
        # Contour Plot
        # ======================================================

        if show_contour:
            st.subheader("Copula CDF Contour Plot")

            fig_contour, ax_contour = plot_2d_function_contour(
                copula_cdf_func,
                title=f"{copula_name} Copula CDF Contour",
                xlabel="u",
                ylabel="v",
                grid_size=int(grid_size),
                levels=int(levels),
                **copula_cdf_params
            )

            st.pyplot(fig_contour)
            plt.close(fig_contour)

        # ======================================================
        # Surface Plot
        # ======================================================

        if show_surface:
            st.subheader("Copula CDF Surface Plot")

            fig_surface, ax_surface = plot_2d_function_surface(
                copula_cdf_func,
                title=f"{copula_name} Copula CDF Surface",
                xlabel="u",
                ylabel="v",
                zlabel="C(u, v)",
                grid_size=int(grid_size),
                **copula_cdf_params
            )

            st.pyplot(fig_surface)
            plt.close(fig_surface)
        
        
        # ======================================================
        # PDF Contour Plot
        # ======================================================
        if show_pdf_contour:
            st.subheader("Copula PDF Contour Plot")

            if copula_pdf_func is None:
                st.info(
                    f"The {copula_name} copula is not absolutely continuous,"
                    "so a usual two-dimensional copula density does not exist."
                )

            else :
                fig_pdf_contour, ax_pdf_contour = plot_2d_function_contour(
                    copula_pdf_func,
                    title=f"{copula_name} Copula PDF Contour",
                    xlabel="u",
                    ylabel="v",
                    grid_size=int(grid_size),
                    levels=int(levels),
                    **copula_pdf_params
                )

                st.pyplot(fig_pdf_contour)
                plt.close(fig_pdf_contour)
        
        
        # ======================================================
        # PDF Surface Plot
        # ======================================================

        if show_pdf_surface:
            st.subheader("Copula PDF Surface Plot")

            if copula_pdf_func is None:
                st.info(
                    f"The {copula_name} copula is not absolutely continuous,"
                    "so a usual two-dimensional copula density does not exist."
                )
            else:
                fig_pdf_surface, ax_pdf_surface = plot_2d_function_surface(
                    copula_pdf_func,
                    title=f"{copula_name} Copula PDF Surface",
                    xlabel="u",
                    ylabel="v",
                    zlabel="c(u,v)",
                    grid_size=int(grid_size),
                    **copula_pdf_params
                )

                st.pyplot(fig_pdf_surface)
                plt.close(fig_pdf_surface)




        st.success("Done.")

    except Exception as e:
        st.error(f"Error: {e}")


