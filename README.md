# Copula Risk Simulation


This project is a Python-based simulation tool for copula models and marginal distributions.
It allows users to generate dependent random samples using different copulas, transform them
with chosen marginal distributions, and visualize the resulting dependence structure.

The project is developed as part of a Computerpraktikum and focuses on stochastic simulation 
copulas, Monte Carlo sampling, and visulaization of multivariate distributions.


## 1. Project Overview


Copula Risk Simulation is a Python project for simulating and visualizing dependent random
vectors with copuas. The main idea is to separate a multivariate distribution into two parts:
the marginal distributions of the individual components and the copula that describes their
dependence structure.

The project implements several copula models, including the independence copula,
comonotonic copula, countermonotonic copula, Gaussian copula, and t copula. It also provides
common marignal distributions such as normal, Student-t, exponential, and uniform
distributions. Copula samples can be transformed into joint samples with user-defined
marginals.

The project also includes visulization tools for scatter plots, contour plots, and 3D surface
plots, which help illustrate different dependenc structures.


## 2. Mathematical Backgroud


The mathematical foundation of this project is Sklar's theorem. It
states that a multivariate distribution can be decomposed into its
marginal distribution and a copula.

For a random vector

$$
X = (X_1,\dots, X_d),
$$

let

$$
F_i(x_i) = P(X_i\leq x_i),\quad i = 1,\dots,d,
$$

be the marignal distribution function of the (i)-th component. Then
the joint distribution function can be written as

$$
C(F_(x_1),\dots,F_d(x_d)).
$$

Here, the marginal distributions (F_i) describe the behavior of the
individual components, while the copula (C) describes the dependence structure
between them.

In the simulation, the project first generates samples from a copula,

$$
U = (U_1,\dots, U_d)\sim C,
$$

where each component satisfies

$$
U_i\sim U(0, 1).
$$

These copula samples are then transformed into samples with the
desired marginal distributions by using inverse distribution
functions:

$$
X_i = F_i^{-1}(U_i).
$$

Thies construction makes it possible to combine different
dependence structures with different marginal distributions. For
example, the same Gaussian copula can be combined wiht normal,
Student - t, exponential, or uniform marginals.



## 3. Features





## 4. Installation





## 5. usage Examples





## 6. Project Structure





## 7. Tests






## 8. Future Exensions