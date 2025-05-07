import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from itertools import combinations_with_replacement
from numpy.polynomial.legendre import Legendre
import random
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from resum.utilities import plotting_utils as plotting
from scipy.integrate import nquad
from math import comb

# Set seeds for reproducibility
np.random.seed(42)         # NumPy seed
random.seed(42)            # Python random seed

class PCEMultiFidelityModelVisualizer:
    def __init__(self, fidelities, parameters, degree, trace=None):
        """
        Initialize the multi-fidelity model visualizer.
        Parameters:
        - basis_matrices (dict): Dictionary of basis matrices for each fidelity level.
          Example: {"lf": basis_matrix_lf, "mf": basis_matrix_mf, "hf": basis_matrix_hf}
        - indices (dict): Dictionary of indices mapping one fidelity level to the next.
          Example: {"mf": indices_mf, "hf": indices_hf}
        - priors (dict): Dictionary of prior configurations for each fidelity level.
          Example: {"lf": {"sigma": 0.5}, "mf": {"sigma": 0.1}, "hf": {"sigma": 0.01}}
        """
        self.fidelities = fidelities
        self.nfidelities = len(fidelities)
        self.feature_labels = list(map(str, parameters.keys()))
        self.degree = degree
        self.x_min = np.array([parameters[k][0] for k in self.feature_labels])
        self.x_max = np.array([parameters[k][1] for k in self.feature_labels])

        self.trace = trace
        if trace==None:
            print("Warring: No trace has been given. Please run \"read_trace(path_to_trace)\"")
        
        self.y_marginalized = None
        #self.get_marginalized()

    def read_trace(self, path_to_trace,version="v1.0"):
        self.trace = az.from_netcdf(f"{path_to_trace}/pce_{version}_trace.nc")

    def normalize_to_minus1_plus1(self,x):
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
    
    def reverse_normalize(self, x_norm):
        return self.x_min + (x_norm + 1) * (self.x_max - self.x_min) / 2
  
    def _generate_basis(self, x_data, degree):
        """
        Generate the multivariate Legendre basis for multi-dimensional inputs.

        Parameters:
        - x_data (ndarray): Input data of shape (n_samples, n_dim).

        Returns:
        - basis_matrix (ndarray): Shape (n_samples, n_terms).
        """
        n_samples, n_dim = x_data.shape
        degree_new = degree
        
        terms = []
        # Generate all combinations of terms up to the given degree
        for deg in range(degree_new + 1):
            for combo in combinations_with_replacement(range(n_dim), deg):
                terms.append(combo)

        # Evaluate each term for all samples
        basis_matrix = np.zeros((n_samples, len(terms)))
        for i, term in enumerate(terms):
            poly = np.prod([Legendre.basis(1)(x_data[:, dim]) for dim in term], axis=0)
            basis_matrix[:, i] = poly
        return basis_matrix, degree_new
    
    def generate_y_pred_samples(self, x_data, include_noise=True):
        """
        Generate high-fidelity prediction samples based on posterior trace.
        Parameters:
        - x_data (ndarray): Input data (e.g., validation or test set).
        - trace: Trace object containing posterior samples from PyMC.

        Returns:
        - y_pred_samples (ndarray): A lit of each predicted fidelity samples (shape: list of (n_samples_total x n_hf_samples)).
        """
        y_pred_samples=[]
        basis_matrix_test,_ = self._generate_basis(x_data,self.degree[0])  # Shape: (n_samples, n_terms_hf)
        coeff_samples = self.trace.posterior[f"coeffs_{self.fidelities[0]}"].values
        coeff_samples_flat = coeff_samples.reshape(-1, coeff_samples.shape[-1]) 
        y_pred = np.dot(coeff_samples_flat, basis_matrix_test.T)
        #softplus = lambda x: np.log(1 + np.exp(x))
        #y_pred = softplus(y_pred)

        if include_noise:
            sigma = self.trace.posterior[f"sigma_{self.fidelities[0]}"].values.flatten()
            noise = np.random.normal(0, sigma[:, None], size=y_pred.shape)
            y_pred += noise

        y_pred_samples.append(y_pred)  # Shape: (n_samples_total, n_lf_samples)

        for i,f in enumerate(self.fidelities[1:]):
            # Extract coefficients from the posterior
            coeff_samples_delta = self.trace.posterior[f"coeffs_delta_{f}"].values  # Shape: (n_chains, n_draws, n_terms_hf)
            coeff_samples_delta_flat = coeff_samples_delta.reshape(-1, coeff_samples_delta.shape[-1])  # Shape: (n_samples_total, n_terms_hf)
            basis_matrix_test,_ = self._generate_basis(x_data,self.degree[i+1])
            delta_pred_samples = np.dot(coeff_samples_delta_flat, basis_matrix_test.T)  # Shape: (n_samples_total, n_hf_samples)
            rho_samples = self.trace.posterior[f"rho_{f}"].values  # Shape: (n_chains, n_draws)
            rho_samples_flat = rho_samples.flatten()  # Shape: (n_samples_total,)
            y_pred = rho_samples_flat[:, None] * y_pred_samples[-1] + delta_pred_samples
            #y_pred = softplus(y_pred)
            if include_noise:
                sigma = self.trace.posterior[f"sigma_{f}"].values.flatten()
                noise = np.random.normal(0, sigma[:, None], size=y_pred.shape)
                y_pred += noise

            # Compute HF predictions
            y_pred_samples.append(y_pred)  # Shape: (n_samples_total, n_hf_samples)
        return y_pred_samples

    def predict(self, x_data, fidelity=1):
        """
        Predict the high-fidelity output for a given input using the model.
        Parameters:
        - x_data (ndarray): Input data (e.g., validation or test set).
        - trace: Trace object containing posterior samples from PyMC.

        Returns:
        - y_pred_samples (ndarray): A list of each predicted fidelity samples (shape: list of (n_samples_total x n_hf_samples)).
        """
        y_pred_samples = self.generate_y_pred_samples(x_data)[fidelity]
        return y_pred_samples
    
    def unnormalized_pdf(self, x, fidelity=1):
        pred = self.predict(x, fidelity)
        return np.maximum(pred, 0)

    def normalized_pdf(self, x, bounds, fidelity):
        norm_factor, _ = nquad(self.unnormalized_pdf, bounds)
        return self.unnormalized_pdf(x) / norm_factor
    
    def likelihood(self, x, fidelity=1):
        pred = self.predict(x, fidelity)
        return np.maximum(pred, 1e-12)  # avoid log(0)

    def log_likelihood(self, x):
        return np.log(self.likelihood(x))

    def validate_mse(self, x_data, y_true, include_noise=False):
        """
        Validate the mean squared error (MSE) of the model.

        Parameters:
        - x_data (ndarray): Input data for prediction.
        - y_true (ndarray): True target values.
        - include_noise (bool): Whether to include noise in predictions.

        Returns:
        - list: Mean Squared Error
        """
        mse = []
        for f in range(self.nfidelities):
            x_data_tmp = self.normalize_to_minus1_plus1(x_data[f])
            y_pred_samples = self.generate_y_pred_samples(x_data_tmp, include_noise=include_noise)[f]  # use highest fidelity
            y_pred_mean = y_pred_samples.mean(axis=0)
            mse.append(np.mean((y_true[f] - y_pred_mean) ** 2))

        return mse

    def plot_validation(self, x_data, y_true):
        """
        Plot the validation data with the uncertainty prediction bands.

        Parameters:
        - x_data (ndarray): Input data (e.g., validation or test set).
        - y_true (ndarray): True high-fidelity target values for validation.
        - trace: Trace object containing posterior samples from PyMC.
        """
        if len(x_data) != self.nfidelities:
            print(f"ERROR: Expected data for {self.nfidelities} fidelities, but got {len(x_data)}.")
            return

        mse = self.validate_mse(x_data,y_true)
        coverage = self.validate_coverage(x_data,y_true)

        nrows = self.nfidelities
        ncols = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 5 * nrows), squeeze=False)

        for f in range(self.nfidelities):
            ax = axes[f][0]
            x_data_tmp = self.normalize_to_minus1_plus1(x_data[f])
            y_pred_samples = self.generate_y_pred_samples(x_data_tmp)[f]
            sample_numbers = np.arange(len(y_true[f]))

            ax.fill_between(
                sample_numbers,
                np.percentile(y_pred_samples, 0.5, axis=0),
                np.percentile(y_pred_samples, 99.5, axis=0),
                color="coral", alpha=0.2, label=r'$\pm 3\sigma$'
            )
            ax.fill_between(
                sample_numbers,
                np.percentile(y_pred_samples, 2.5, axis=0),
                np.percentile(y_pred_samples, 97.5, axis=0),
                color="yellow", alpha=0.2, label=r'$\pm 2\sigma$'
            )
            ax.fill_between(
                sample_numbers,
                np.percentile(y_pred_samples, 16, axis=0),
                np.percentile(y_pred_samples, 84, axis=0),
                color="green", alpha=0.2, label=r'$\pm 1\sigma$'
            )
            ax.scatter(sample_numbers, y_true[f], marker='.',s=5, color="black", label=f"{self.fidelities[f]} Validation Data")

            ax.set_xlabel(f"{self.fidelities[f]} Simulation Trial Number")
            ax.set_ylabel(r"Predicted $y^{("+f"{self.fidelities[f]}"+")}$")
            text = f"MSE: {mse[f]:.5f} $\pm1\sigma$: {coverage[self.fidelities[f]][1]:.1f}%  $\pm3\sigma$: {coverage[self.fidelities[f]][2]:.1f}%  $\pm3\sigma$: {coverage[self.fidelities[f]][3]:.1f}%"
            plotting.place_text_corner(ax, text, fontsize=8, bbox=dict(edgecolor='gray', facecolor='none', linewidth=0.5))
            

        legend_elements = [
            Line2D([0], [0], marker='.', color='black', linestyle='None', label='Data'),
            Line2D([0], [0], marker='.', color='white', linestyle='None', label='Model prediction'),
            mpatches.Patch(color='green', alpha=0.2, label=r'$\pm 1\sigma$'),
            mpatches.Patch(color='yellow', alpha=0.2, label=r'$\pm 2\sigma$'),
            mpatches.Patch(color='coral', alpha=0.2, label=r'$\pm 3\sigma$')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), fontsize='medium', frameon=False, bbox_to_anchor=(0.5, 1.05))
        plt.tight_layout()
        plt.show()
        return fig

    def validate_coverage(self, x_data, y_true):
        """
        Validate the coverage of the model for 1, 2, and 3 sigma intervals.

        Parameters:
        - y_true (ndarray): True high-fidelity target values for validation.
        - y_hf_pred_samples (ndarray): Posterior predictive samples for high-fidelity predictions.

        Returns:
        - dict: Percentages of validation data within 1, 2, and 3 sigma intervals.
        """
        coverage={}
        for ix in range(len(x_data)):
            x_data_tmp = self.normalize_to_minus1_plus1(x_data[ix])
            y_hf_pred_samples = self.generate_y_pred_samples(x_data_tmp)[ix]
            y_true_tmp = y_true[ix]
            counters = {1: 0, 2: 0, 3: 0}

            # Calculate percentile intervals for the posterior samples
            percentiles = {
                1: (np.percentile(y_hf_pred_samples, 16, axis=0), np.percentile(y_hf_pred_samples, 84, axis=0)),
                2: (np.percentile(y_hf_pred_samples, 2.5, axis=0), np.percentile(y_hf_pred_samples, 97.5, axis=0)),
                3: (np.percentile(y_hf_pred_samples, 0.5, axis=0), np.percentile(y_hf_pred_samples, 99.5, axis=0)),
            }

            # Count the number of y_true points within each interval
            for i, y in enumerate(y_true_tmp):
                for sigma in [1, 2, 3]:
                    low, high = percentiles[sigma]
                    if low[i] <= y <= high[i]:
                        counters[sigma] += 1

            # Calculate percentages
            coverage[self.fidelities[ix]]={sigma: (counters[sigma] / len(y_true_tmp)) * 100 for sigma in [1, 2, 3]}
        return coverage
    
    def get_marginalized(self, grid_steps=20):
            def reverse_norm(x_norm, x_min, x_max):
                return x_min + (x_norm + 1) * (x_max - x_min) / 2

            x_grid_norm_list = []
            self.x_grid = []
            grid_steps_list = []
            for i in range(len(self.x_min)):
                grid_steps_list.append(grid_steps)
                arr = np.linspace(-1., 1., grid_steps)
                x_grid_norm_list.append(arr)
                self.x_grid.append(reverse_norm(x_grid_norm_list[-1],self.x_min[i],self.x_max[i]))

            mesh = np.meshgrid(*x_grid_norm_list, indexing='ij')
            x_grid = np.column_stack([x.flatten() for x in mesh])  # shape: (m, 4)

            

            y = self.generate_y_pred_samples(x_grid) # is a list of shape (n_posterior_draws, n_data_samples)

            self.y_marginalized = []
            for f in range(self.nfidelities):
                self.y_marginalized.append([])
                
                for ix in range(len(self.x_min)):
                    y_grid = y[f].reshape(len(y[f]),*grid_steps_list)
                    # Define axes to marginalize (all axes except the kept one)
                    all_axes = list(range(1, y_grid.ndim))  # skip the draw axis (axis 0)
                    marg_axes = tuple(ax for ax in all_axes if ax != (ix + 1))
                    self.y_marginalized[-1].append(np.mean(y_grid, axis=marg_axes))

    def plot_marginalized(self,x_data=None, y_data=None, grid_steps=20):
        if self.y_marginalized is None:
            self.get_marginalized(grid_steps=grid_steps)
        nrows = self.nfidelities
        ncols = len(self.x_max)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for f in range(self.nfidelities):
            for keep_axis in range(len(self.x_max)):
                ax = axes[f][keep_axis]

                y_mean = np.percentile(self.y_marginalized[f][keep_axis], 50., axis=0)
                y_1sigma_low = np.percentile(self.y_marginalized[f][keep_axis], 16., axis=0)
                y_1sigma_high = np.percentile(self.y_marginalized[f][keep_axis], 84., axis=0)
                y_2sigma_low = np.percentile(self.y_marginalized[f][keep_axis], 2.5, axis=0)
                y_2sigma_high = np.percentile(self.y_marginalized[f][keep_axis], 97.5, axis=0)
                y_3sigma_low = np.percentile(self.y_marginalized[f][keep_axis], 0.5, axis=0)
                y_3sigma_high = np.percentile(self.y_marginalized[f][keep_axis], 99.5, axis=0)

                ax.fill_between(
                    self.x_grid[keep_axis], y_3sigma_low, y_3sigma_high,
                    color="coral", alpha=0.2, label=r'$\pm 3\sigma$'
                )
                ax.fill_between(
                    self.x_grid[keep_axis], y_2sigma_low, y_2sigma_high,
                    color="yellow", alpha=0.2, label=r'$\pm 2\sigma$'
                )
                ax.fill_between(
                    self.x_grid[keep_axis], y_1sigma_low, y_1sigma_high,
                    color="green", alpha=0.2, label=r'$\pm 1\sigma$'
                )
                ax.plot(self.x_grid[keep_axis], y_mean, color="black", label="Model")
                if x_data is not None and y_data is not None:
                    x, y, _, _ = self.get_marginalized_single_draw(x_data[f], y_data[f], keep_axis=keep_axis, grid_steps=grid_steps)
                    ax.scatter(x, y, marker='.', color="black", label="Data")
                    w_data=False

                #y_ax=np.nan_to_num(y, nan=0.0)
                #ax.set_ylim(np.min(y_ax)*0.8,np.max(y_ax)*1.2)
                if f==0:
                    ax.set_ylim(0.13,0.23)
                elif f==1:
                    ax.set_ylim(-0.01,0.03)
                ax.set_xlabel(f'{self.feature_labels[keep_axis]}')
                str_tmp = f"{self.fidelities[f]}"
                ax.set_ylabel('Marginalized predicted $y^{('+str_tmp+')}$')


        # Create custom legend handles
        if x_data is None:
            legend_elements = [
                Line2D([0], [0], color='black', lw=2, label='Model prediction'),
                mpatches.Patch(color='green', alpha=0.2, label=r'$\pm 1\sigma$'),
                mpatches.Patch(color='yellow', alpha=0.2, label=r'$\pm 2\sigma$'),
                mpatches.Patch(color='coral', alpha=0.2, label=r'$\pm 3\sigma$')
            ]
        else:
            legend_elements = [
                Line2D([0], [0], marker='.', color='black', linestyle='None', label='Data'),
                Line2D([0], [0], color='black', lw=2, label='Model prediction'),
                mpatches.Patch(color='green', alpha=0.2, label=r'$\pm 1\sigma$'),
                mpatches.Patch(color='yellow', alpha=0.2, label=r'$\pm 2\sigma$'),
                mpatches.Patch(color='coral', alpha=0.2, label=r'$\pm 3\sigma$')
            ]


        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), fontsize='medium', frameon=False, bbox_to_anchor=(0.5, 1.05))
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
        plt.show()
        return fig

    def get_marginalized_random(self, x_data, keep_axis=0, grid_steps=10):

        for f in self.fidelities:
            x_data_normalized = self.normalize_to_minus1_plus1(x_data)
        
        #    y_hf is assumed to have shape (n_posterior_draws, n_samples)
        y_hf = self.generate_y_pred_samples(x_data_normalized)[-1]
        
        # Extract the keep_axis values from the random inputs and reverse the normalization.
        x_keep = x_data[:, keep_axis]

        # Define bins along the kept axis (using the unnormalized space for plotting).
        bin_edges = np.linspace(self.x_min[keep_axis], self.x_max[keep_axis], grid_steps + 1)
        # Compute the bin centers for plotting.
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # For each bin, compute the average predicted y for each posterior draw.
        n_draws = y_hf.shape[0]
        binned_means = np.empty((n_draws, grid_steps))
        
        # Loop over each bin to compute means.
        for i in range(grid_steps):
            # Define a mask for the samples falling in the current bin.
            # For all but the last bin, use a half-open interval; include the right edge only for the last bin.
            if i < grid_steps - 1:
                bin_mask = (x_keep >= bin_edges[i]) & (x_keep < bin_edges[i+1])
            else:
                bin_mask = (x_keep >= bin_edges[i]) & (x_keep <= bin_edges[i+1])
            
            # Check if there are any samples in the bin.
            if np.sum(bin_mask) == 0:
                # If no samples fall in the bin, assign a NaN value.
                binned_means[:, i] = np.nan
            else:
                # For each posterior draw, average the predictions of the samples in the bin.
                # y_hf has shape (n_draws, n_samples), so for each draw we average over the masked indices.
                binned_means[:, i] = np.mean(y_hf[:, bin_mask], axis=1)
        
        # Compute the percentiles across posterior draws for each bin.
        y_hf_mean       = np.nanpercentile(binned_means, 50, axis=0)
        y_hf_1sigma_low = np.nanpercentile(binned_means, 16, axis=0)
        y_hf_1sigma_high= np.nanpercentile(binned_means, 84, axis=0)

        return y_hf_mean, y_hf_1sigma_low, y_hf_1sigma_high
    
    def get_marginalized_single_draw(self, x_data, y_data, keep_axis, grid_steps=25):
        """
        Marginalizes predictions over all but one feature using random sampling when only one 
        prediction is available per sample (y_hf has shape (n_samples, 1)).
        """

        x_keep = x_data[:, keep_axis]

        # 4. Define bins along the kept axis in the original scale.
        bin_edges = np.linspace(self.x_min[keep_axis], self.x_max[keep_axis], grid_steps + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # 5. For each bin, compute the median and 1σ percentiles (16th and 84th) from the samples in the bin.
        medians = np.empty(grid_steps)
        lower_vals = np.empty(grid_steps)
        upper_vals = np.empty(grid_steps)
        for i in range(grid_steps):
            # Use a half-open interval except for the last bin.
            if i < grid_steps - 1:
                mask = (x_keep >= bin_edges[i]) & (x_keep < bin_edges[i+1])
            else:
                mask = (x_keep >= bin_edges[i]) & (x_keep <= bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_values = y_data[mask]
                medians[i] = np.median(bin_values)
                lower_vals[i] = np.percentile(bin_values, 16)
                upper_vals[i] = np.percentile(bin_values, 84)
            else:
                medians[i] = np.nan
                lower_vals[i] = np.nan
                upper_vals[i] = np.nan

        # Compute errors for plotting (errorbars represent the distance from the median to the percentiles)
        lower_error = medians - lower_vals
        upper_error = upper_vals - medians
        
        return bin_centers, medians, lower_error, upper_error
    
