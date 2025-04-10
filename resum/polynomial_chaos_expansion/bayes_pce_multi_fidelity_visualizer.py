import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
from itertools import combinations_with_replacement
from numpy.polynomial.legendre import Legendre
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from itertools import product, combinations
import random

# Set seeds for reproducibility
np.random.seed(42)         # NumPy seed
random.seed(42)            # Python random seed


class PCEMultiFidelityModelVisualizer:
    def __init__(self, fidelities, parameters, degree, trace=None):
        """
        Initialize the multi-fidelity model.

        Parameters:
        - basis_matrices (dict): Dictionary of basis matrices for each fidelity level.
          Example: {"lf": basis_matrix_lf, "mf": basis_matrix_mf, "hf": basis_matrix_hf}
        - indices (dict): Dictionary of indices mapping one fidelity level to the next.
          Example: {"mf": indices_mf, "hf": indices_hf}
        - priors (dict): Dictionary of prior configurations for each fidelity level.
          Example: {"lf": {"sigma": 0.5}, "mf": {"sigma": 0.1}, "hf": {"sigma": 0.01}}
        """
        self.fidelities = fidelities
        self.feature_labels = list(map(str, parameters.keys()))
        self.degree = degree
        self.x_min = np.array([parameters[k][0] for k in self.feature_labels])
        self.x_max = np.array([parameters[k][1] for k in self.feature_labels])

        self.trace = trace
        if trace==None:
            print("Warring: No trace has been given. Please run \"read_trace(path_to_trace)\"")

    def read_trace(self, path_to_trace,version="v1.0"):
        self.trace = az.from_netcdf(f"{path_to_trace}/pce_{version}_trace.nc")

    def normalize_to_minus1_plus1(self,x):
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
    
    def reverse_normalize(self, x_norm):
        return self.x_min + (x_norm + 1) * (self.x_max - self.x_min) / 2

    def _generate_basis(self, x_data):
        """
        Generate the multivariate Legendre basis for multi-dimensional inputs.

        Parameters:
        - x_data (ndarray): Input data of shape (n_samples, n_dim).

        Returns:
        - basis_matrix (ndarray): Shape (n_samples, n_terms).
        """
            
        n_samples, n_dim = x_data.shape
        terms = []
        # Generate all combinations of terms up to the given degree
        for deg in range(self.degree + 1):
            for combo in combinations_with_replacement(range(n_dim), deg):
                terms.append(combo)

        # Evaluate each term for all samples
        basis_matrix = np.zeros((n_samples, len(terms)))
        for i, term in enumerate(terms):
            poly = np.prod([Legendre.basis(1)(x_data[:, dim]) for dim in term], axis=0)
            basis_matrix[:, i] = poly
        return basis_matrix
    
    def generate_y_hf_pred_samples(self, x_data):
        """
        Generate high-fidelity prediction samples based on posterior trace.

        Parameters:
        - x_data (ndarray): Input data (e.g., validation or test set).
        - trace: Trace object containing posterior samples from PyMC.

        Returns:
        - y_hf_pred_samples (ndarray): Predicted high-fidelity samples (shape: n_samples_total x n_hf_samples).
        """
        basis_matrix_test = self._generate_basis(x_data)  # Shape: (n_samples, n_terms_hf)

        coeff_samples = self.trace.posterior[f"coeffs_{self.fidelities[0]}"].values  # Shape: (n_chains, n_draws, n_terms_hf)
        coeff_samples_flat = coeff_samples.reshape(-1, coeff_samples.shape[-1])  # Shape: (n_samples_total, n_terms_hf)
        y_pred_samples = np.dot(coeff_samples_flat, basis_matrix_test.T)  # Shape: (n_samples_total, n_lf_samples)

        for fidelity in self.fidelities[1:]:

            # Extract coefficients from the posterior
            coeff_samples_delta = self.trace.posterior[f"coeffs_delta_{fidelity}"].values  # Shape: (n_chains, n_draws, n_terms_hf)
            coeff_samples_delta_flat = coeff_samples_delta.reshape(-1, coeff_samples_delta.shape[-1])  # Shape: (n_samples_total, n_terms_hf)
            delta_pred_samples = np.dot(coeff_samples_delta_flat, basis_matrix_test.T)  # Shape: (n_samples_total, n_hf_samples)
            rho_samples = self.trace.posterior[f"rho_{fidelity}"].values  # Shape: (n_chains, n_draws)
            rho_samples_flat = rho_samples.flatten()  # Shape: (n_samples_total,)
            
            # Compute HF predictions
            y_pred_samples = rho_samples_flat[:, None] * y_pred_samples + delta_pred_samples  # Shape: (n_samples_total, n_hf_samples)

        return y_pred_samples
    
    def get_marginalized(self, keep_axis=0, grid_steps=10):
            x_grid_list = []
            for i in range(len(self.x_min)):
                arr = np.linspace(-1., 1., grid_steps)
                x_grid_list.append(arr)

            mesh = np.meshgrid(*x_grid_list, indexing='ij')
            x_data = np.column_stack([x.flatten() for x in mesh])  # shape: (m, 4)

            y_hf = self.generate_y_hf_pred_samples(x_data) # y_hf has shape (n_posterior_draws, n_data_samples)
            y_hf_mean = np.percentile(y_hf, 50., axis=0)
            y_hf_1sigma_low = np.percentile(y_hf, 16., axis=0)
            y_hf_1sigma_low_grid = y_hf_1sigma_low.reshape(grid_steps, grid_steps, grid_steps, grid_steps)
            y_hf_1sigma_high = np.percentile(y_hf, 84., axis=0)
            y_hf_1sigma_high_grid = y_hf_1sigma_high.reshape(grid_steps, grid_steps, grid_steps, grid_steps)
            y_hf_2sigma_low = np.percentile(y_hf, 2.5, axis=0)
            y_hf_2sigma_low_grid = y_hf_2sigma_low.reshape(grid_steps, grid_steps, grid_steps, grid_steps)
            y_hf_2sigma_high = np.percentile(y_hf, 97.5, axis=0)
            y_hf_2sigma_high_grid = y_hf_2sigma_high.reshape(grid_steps, grid_steps, grid_steps, grid_steps)
            y_hf_3sigma_low = np.percentile(y_hf, 0.5, axis=0)
            y_hf_3sigma_low_grid = y_hf_3sigma_low.reshape(grid_steps, grid_steps, grid_steps, grid_steps)
            y_hf_3sigma_high = np.percentile(y_hf, 99.5, axis=0)
            y_hf_3sigma_high_grid = y_hf_3sigma_high.reshape(grid_steps, grid_steps, grid_steps, grid_steps)
            
            y_grid = y_hf_mean.reshape(grid_steps, grid_steps, grid_steps, grid_steps)
            all_grid_axes = list(range(len(self.x_min)))
            marginalize_axes = tuple(ax for ax in all_grid_axes if ax != keep_axis)

            # Now, marginalize over the 4th axis (axis 3 in zero-indexing):
            y_marginalized = np.mean(y_grid, axis=marginalize_axes)
            y_hf_1sigma_low_grid = np.mean(y_hf_1sigma_low_grid, axis=marginalize_axes)
            y_hf_1sigma_high_grid = np.mean(y_hf_1sigma_high_grid, axis=marginalize_axes)
            y_hf_2sigma_low_grid = np.mean(y_hf_2sigma_low_grid, axis=marginalize_axes)
            y_hf_2sigma_high_grid = np.mean(y_hf_2sigma_high_grid, axis=marginalize_axes)
            y_hf_3sigma_low_grid = np.mean(y_hf_3sigma_low_grid, axis=marginalize_axes)
            y_hf_3sigma_high_grid = np.mean(y_hf_3sigma_high_grid, axis=marginalize_axes)
            
            def reverse_norm(x_norm, x_min, x_max):
                return x_min + (x_norm + 1) * (x_max - x_min) / 2
            
            x_rescaled = reverse_norm(x_grid_list[keep_axis],self.x_min[keep_axis],self.x_max[keep_axis])
            
            # Plot the marginalized predictions vs. the chosen feature.
            plt.figure(figsize=(10, 6))
            plt.fill_between(
                x_rescaled,
                y_hf_3sigma_low_grid,
                y_hf_3sigma_high_grid,
                color="coral", alpha=0.2, label=r'$\pm 3\sigma$'
            )

            plt.fill_between(
                x_rescaled,
                y_hf_2sigma_low_grid,
                y_hf_2sigma_high_grid,
                color="yellow", alpha=0.2, label=r'$\pm 2\sigma$'
            )
            plt.fill_between(
                x_rescaled,
                y_hf_1sigma_low_grid,
                y_hf_1sigma_high_grid,
                color="green", alpha=0.2, label=r'$\pm 1\sigma$'
            )
            plt.plot(x_rescaled, y_marginalized, marker='',color="black")

            plt.xlabel(f'{self.feature_labels[keep_axis]}')
            plt.ylabel('Marginalized predicted y (HF)')
            plt.title(f'Predicted y_hf vs. {self.feature_labels[keep_axis]} (marginalized)')
            plt.show()
