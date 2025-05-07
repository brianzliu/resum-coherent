import numpy as np
import GPy
import copy
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans

class Normalizer:
    def __init__(self):
        self.scaler_x = None
        self.scaler_y = None

    def fit(self, trainings_data):
        """
        trainings_data: dict like {0: (X_lf, Y_lf), 1: (X_hf, Y_hf)}
        Fits one global scaler for X and one for Y across all fidelities.
        """
        # Collect all X and Y
        x_list = []
        y_list = []
        for key in trainings_data.keys():
            x, y = trainings_data[key]
            x = np.atleast_2d(x)
            y = np.atleast_2d(y).reshape(-1, 1)  # Ensure (N, 1)
            x_list.append(x)
            y_list.append(y)

        # Stack together and fit scalers
        x_all = np.vstack(x_list)
        y_all = np.vstack(y_list)

        self.scaler_x = StandardScaler().fit(x_all)
        self.scaler_y = StandardScaler().fit(y_all)

    def transform_x(self, x):
        x = np.atleast_2d(x)
        return self.scaler_x.transform(x)

    def transform_y(self, y):
        y = np.atleast_2d(y).reshape(-1, 1)
        return self.scaler_y.transform(y)

    def transform_noise(self, noise):
        """Normalize noise (standard deviation) according to y-scaling."""
        sigma_y = self.scaler_y.scale_[0]  # Only one output dimension
        return noise / sigma_y

    def inverse_transform_x(self, x_norm):
        """Undo normalization for predictions (mean values)."""
        return self.scaler_x.inverse_transform(np.atleast_2d(x_norm))
    
    def inverse_transform_y(self, y_norm):
        """Undo normalization for predictions (mean values)."""
        return self.scaler_y.inverse_transform(np.atleast_2d(y_norm))

    def inverse_transform_noise(self, noise_norm):
        """Undo normalization for predictive noise (std)."""
        sigma_y = self.scaler_y.scale_[0]
        return noise_norm * sigma_y

    def inverse_transform_variance(self, var_norm):
        """Undo normalization for predictive variance."""
        sigma_y = self.scaler_y.scale_[0]
        return var_norm * (sigma_y ** 2)
    
class DeepGPModel():
    def __init__(self, trainings_data, normalize_data=True):
        self.trainings_data = copy.deepcopy(trainings_data)
        self.fidelities = list(self.trainings_data.keys())
        self.nfidelities = len(self.fidelities)
        self.normalize_data = normalize_data
        self.X_scalers = {}
        self.Y_scalers = {}
        self.gp1_models = []  # GPs for X -> H
        self.gp2_model = None  # GP for H -> Y
        self.gp_delta = None  # GP for correction
        self.rho = 1.  # Default scaling

    def set_trainings_data(self, trainings_data):
        self.trainings_data = copy.deepcopy(trainings_data)

    def normalize_xy(self, x_list, y_list):
        # First, concatenate all x and y across fidelities
        x_all = np.vstack(x_list)
        y_all = np.vstack(y_list)

        # Fit one scaler on all data
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        scaler_x.fit(x_all)
        scaler_y.fit(y_all)

        # Save the scalers (only one for all fidelities)
        self.X_scaler = scaler_x
        self.Y_scaler = scaler_y

        # Now transform each fidelity individually with the same scaler
        x_list_norm = [scaler_x.transform(x) for x in x_list]
        y_list_norm = [scaler_y.transform(y) for y in y_list]

        return x_list_norm, y_list_norm

    def build_model(self, n_inducing=100, n_restarts=10, noise_dict=None):
        # --- Check that LF and HF data are available ---
        assert 0 in self.trainings_data and 1 in self.trainings_data, "Training data must contain fidelity 0 (LF) and 1 (HF)."

        # --- Extract training data ---
        X_lf, Y_lf = self.trainings_data[0]
        X_hf, Y_hf = self.trainings_data[1]

        # --- Ensure correct array shapes ---
        X_lf = np.atleast_2d(X_lf)
        Y_lf = np.atleast_2d(Y_lf).reshape(-1, 1)  # Ensure (N, 1)
        X_hf = np.atleast_2d(X_hf)
        Y_hf = np.atleast_2d(Y_hf).reshape(-1, 1)

        # --- Initialize normalizer and fit it ---
        self.normalizer = Normalizer()
        self.normalizer.fit(self.trainings_data)

        # --- Normalize training data ---
        if self.normalize_data:
            X_lf = self.normalizer.transform_x(X_lf)
            Y_lf = self.normalizer.transform_y(Y_lf)

            X_hf = self.normalizer.transform_x(X_hf)
            Y_hf = self.normalizer.transform_y(Y_hf)

        input_dim = X_lf.shape[1]

        # --- Build single GP: X -> Y_lf ---
        n_inducing = min(100, X_lf.shape[0])

        # Build GP
        kernel = (GPy.kern.RBF(input_dim, ARD=True) +
                GPy.kern.Bias(input_dim) +
                GPy.kern.Matern32(input_dim, ARD=True))

        self.deepgp_lf = GPy.models.SparseGPRegression(X_lf, Y_lf, kernel=kernel, num_inducing=n_inducing)

        # Smart inducing points (k-means)
        kmeans = KMeans(n_clusters=n_inducing, random_state=42).fit(X_lf)
        self.deepgp_lf.Z[:] = kmeans.cluster_centers_

        # Noise
        
        if noise_dict is not None:
            # --- Normalize noise (if you have predictive stds etc.) ---
            noise_lf = noise_dict.get(0,0.05)
            noise_hf = noise_dict.get(1,1e-8)
            print(f"Deep GP LF noise fixed to {noise_lf}")
            if self.normalize_data:
                noise_lf = self.normalizer.transform_noise(noise_lf)
                noise_hf = self.normalizer.transform_noise(noise_hf)
            self.deepgp_lf.Gaussian_noise.variance.fix(noise_lf)
            

        # Optimize
        self.deepgp_lf.optimize(messages=True, max_iters=1000)
        self.deepgp_lf.optimize_restarts(num_restarts=n_restarts, verbose=True)

        lf_pred_hf, lf_var = self.deepgp_lf.predict(X_hf)

        # Optimize rho before training correction GP
        print("Optimizing rho...")
        def mse_rho(rho_array):
            rho_val = rho_array[0]
            print(rho_val)
            return np.mean((Y_hf - rho_val * lf_pred_hf)**2)

        res = minimize(mse_rho, x0=[self.rho], bounds=[(1e-5, 10.0)])
        self.rho = res.x[0]
        print(f"Optimized rho: {self.rho:.5f}")


        # Compute residuals after rho optimization
        residuals = Y_hf - self.rho * lf_pred_hf

        # Build Correction GP
        print("Building correction GP...")
        kernel_delta = GPy.kern.Matern32(1, ARD=True)
        #kernel_delta = GPy.kern.RBF(1, ARD=True)
        self.gp_delta = GPy.models.GPRegression(X_hf, residuals, kernel_delta)
        if noise_dict is not None:
            self.gp_delta.Gaussian_noise.variance.fix(noise_hf)
            #self.gp_delta.Gaussian_noise.variance.constrain_bounded(1e-8, 1e-3)
        else:
            self.gp_delta.Gaussian_noise.variance.constrain_bounded(1e-8, 1e-3)
        
        self.gp_delta.kern.lengthscale.constrain_bounded(0.5, 10.0)
        self.gp_delta.kern.variance.constrain_bounded(0.1, 50.0)
        
        self.gp_delta.optimize(messages=True, max_iters=500)
        for _ in range(n_restarts):
            self.gp_delta.optimize(messages=True, max_iters=500)
        print(self.gp_delta)
        print(self.gp_delta.kern.lengthscale.values)
        print("Deep Multi-Fidelity Correction Model built and optimized.")

        return self

    def predict(self, test_data):
        predictions = {}

        for fidelity, (X_test, _) in test_data.items():

            # --- Ensure correct array shapes ---
            X = np.atleast_2d(X_test)

            if self.normalize_data:
                # --- Normalize training data ---
                X = self.normalizer.transform_x(X)

            # Predict LF output
            lf_pred, lf_var = self.deepgp_lf.predict(X)

            if fidelity == 0:
                y_pred = lf_pred
                y_var = lf_var
            else:
                delta_pred, delta_var = self.gp_delta.predict(X)
                y_pred = self.rho * lf_pred + delta_pred
                y_var = (self.rho ** 2) * lf_var + delta_var #+ 2 * self.rho * self.empirical_covariance

            if self.normalize_data:
                y_pred=self.normalizer.inverse_transform_y(y_pred)
                y_var = self.normalizer.inverse_transform_variance(y_var)   

            predictions[fidelity] = (y_pred, y_var)

        return predictions
    

    def plot_prediction_trace(self, test_data, predictions, title_prefix=""):
        """
        Plots prediction mean with ±1σ, ±2σ, ±3σ bands and true Y over data points.
        Args:
            test_data: dict of {fidelity: (X_test, Y_test)}
            predictions: dict of {fidelity: (Y_pred, Y_var)}
            title_prefix: optional string to prepend to plot titles
        """

        n_fidelities = len(test_data)

        # Find the maximum length for a common x-axis
        max_n_points = max(len(Y[1]) for Y in test_data.values())
        x_axis = np.arange(max_n_points)

        fig, axes = plt.subplots(n_fidelities, 1, figsize=(12, 5 * n_fidelities), sharex=True)

        if n_fidelities == 1:
            axes = [axes]

        for idx, (fidelity, (X_test, Y_true)) in enumerate(test_data.items()):
            Y_pred, Y_var = predictions.get(fidelity, (None, None))

            if Y_pred is None or Y_var is None:
                print(f"Warning: No prediction available for fidelity {fidelity}. Skipping.")
                continue

            # Ensure arrays
            X_test = np.asarray(X_test)
            Y_true = np.asarray(Y_true).flatten()
            Y_pred = np.asarray(Y_pred).flatten()
            Y_std = np.sqrt(np.asarray(Y_var).flatten())

            # Match arrays to common size
            n_points = len(Y_pred)
            if n_points < max_n_points:
                # Pad with NaNs if necessary
                pad_width = max_n_points - n_points
                Y_pred = np.pad(Y_pred, (0, pad_width), constant_values=np.nan)
                Y_std = np.pad(Y_std, (0, pad_width), constant_values=np.nan)
                Y_true = np.pad(Y_true, (0, pad_width), constant_values=np.nan)
            else:
                # Slice if somehow too long
                Y_pred = Y_pred[:max_n_points]
                Y_std = Y_std[:max_n_points]
                Y_true = Y_true[:max_n_points]

            ax = axes[idx]

            # Plot prediction mean
            ax.plot(x_axis, Y_pred, color='black', linewidth=0.5, label="Prediction mean")

            # Plot uncertainty bands
            for sigma, color, alpha in zip([1, 2, 3], ['green', 'yellow', 'coral'], [0.3, 0.2, 0.1]):
                lower = Y_pred - sigma * Y_std
                upper = Y_pred + sigma * Y_std
                ax.fill_between(
                    x_axis,
                    lower,
                    upper,
                    color=color,
                    alpha=alpha,
                    label=f"±{sigma}σ" if sigma == 1 else None  # Only label ±1σ
                )

            # Plot true Y
            ax.plot(x_axis, Y_true, 'k.', markersize=6, label="True Y")

            # MSE and coverage calculations (excluding padded NaNs)
            valid = ~np.isnan(Y_true) & ~np.isnan(Y_pred)
            mse = np.mean((Y_pred[valid] - Y_true[valid]) ** 2)

            residuals = np.abs(Y_pred[valid] - Y_true[valid])
            Y_std_valid = Y_std[valid]

            total_samples = np.sum(valid)
            counts = [np.sum(residuals <= k * Y_std_valid) for k in [1, 2, 3]]
            ratios = [count * 100. / total_samples for count in counts]

            print(f"Fidelity {fidelity} Results:")
            print(f"  MSE          : {mse:.4e}")
            print(f"  Coverage ±1σ : {ratios[0]:.1f}%")
            print(f"  Coverage ±2σ : {ratios[1]:.1f}%")
            print(f"  Coverage ±3σ : {ratios[2]:.1f}%")

            ax.set_ylabel("Predicted Y")
            ax.set_title(f"{title_prefix} Fidelity {fidelity}")
            ax.grid(True)

            # Clean up legend (remove duplicates)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        axes[-1].set_xlabel("Data point index")  # Set xlabel only on bottom plot
        plt.tight_layout()
        plt.show()

        return fig
    
    def get_marginalized_predictions(self, x_mins, x_maxs, grid_steps=40, marginal_steps=30, random_seed=42):
        """
        Computes marginalized predictions for each feature separately by averaging predictions
        over all other dimensions.
        """
        np.random.seed(random_seed)
        self.predictions_marginalized = {}
        fidelities = list(self.trainings_data.keys())


        for f in fidelities:
            X_train, _ = self.trainings_data[f]
            X_train = np.atleast_2d(X_train)

        self.x_grid = []
        self.y_marginalized = {}

        for f in fidelities:
            X_train = np.atleast_2d(self.trainings_data[f][0])
            input_dim = X_train.shape[1]

            marginalized_outputs = []

            for dim in range(input_dim):
                # Grid for dimension of interest
                x_linspace = np.linspace(x_mins[dim], x_maxs[dim], grid_steps)
                preds_along_grid = []

                for val in x_linspace:
                    # Random samples for other dimensions
                    X_query = np.random.uniform(x_mins, x_maxs, size=(marginal_steps, input_dim))
                    X_query[:, dim] = val  # Fix the feature

                    # Predict
                    test_dict = {f: (X_query, np.zeros((marginal_steps, 1)))}
                    preds = self.predict(test_dict)

                    y_pred, y_var = preds[f]
                    std_pred = np.sqrt(y_var)

                    # Sample from the GP predictive distribution
                    y_sampled = y_pred.flatten() + np.random.randn(marginal_steps) * std_pred.flatten()

                    preds_along_grid.append(y_sampled)


                marginalized_outputs.append(np.array(preds_along_grid))  # (grid_steps, marginal_steps)

            self.x_grid = [np.linspace(x_mins[d], x_maxs[d], grid_steps) for d in range(input_dim)]
            self.y_marginalized[f] = marginalized_outputs

        return self

    def plot_marginalized_predictions(self, x_data=None, y_data=None):
        """
        Plots the marginalized predictions (mean ± uncertainty) for each feature.
        """
        if not hasattr(self, 'y_marginalized'):
            raise ValueError("Marginalized predictions not found. Run get_marginalized_predictions() first.")

        n_fidelities = len(self.y_marginalized)
        n_features = len(self.x_grid)
        fig, axes = plt.subplots(nrows=n_fidelities, ncols=n_features, figsize=(5 * n_features, 4 * n_fidelities), squeeze=False)

        for f_idx, (fidelity, marginals) in enumerate(self.y_marginalized.items()):
            for dim in range(n_features):
                ax = axes[f_idx][dim]

                y_samples = marginals[dim]  # Shape: (grid_steps, marginal_steps)
                x_vals = self.x_grid[dim]

                y_mean = np.percentile(y_samples, 50., axis=1)
                y_1sigma_low = np.percentile(y_samples, 16., axis=1)
                y_1sigma_high = np.percentile(y_samples, 84., axis=1)
                y_2sigma_low = np.percentile(y_samples, 2.5, axis=1)
                y_2sigma_high = np.percentile(y_samples, 97.5, axis=1)
                y_3sigma_low = np.percentile(y_samples, 0.5, axis=1)
                y_3sigma_high = np.percentile(y_samples, 99.5, axis=1)

                ax.fill_between(x_vals, y_3sigma_low, y_3sigma_high, color="coral", alpha=0.2, label=r'$\pm 3\sigma$')
                ax.fill_between(x_vals, y_2sigma_low, y_2sigma_high, color="yellow", alpha=0.3, label=r'$\pm 2\sigma$')
                ax.fill_between(x_vals, y_1sigma_low, y_1sigma_high, color="green", alpha=0.3, label=r'$\pm 1\sigma$')

                ax.plot(x_vals, y_mean, color="black", lw=2, label="Model")

                if x_data is not None and y_data is not None:
                    if fidelity in x_data and fidelity in y_data:
                        Xd, Yd = x_data[fidelity]
                        Xd = np.atleast_2d(Xd)
                        Yd = np.atleast_2d(Yd)
                        if Xd.shape[0] == Yd.shape[0]:
                            ax.scatter(Xd[:, dim], Yd.flatten(), color="blue", s=15, alpha=0.7, label="Data")

                ax.set_xlabel(f"Feature {dim}", fontsize=12)
                ax.set_ylabel(f"Predicted Y", fontsize=12)
                ax.set_title(f"Fidelity {fidelity} - Feature {dim}", fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.tick_params(axis='both', which='major', labelsize=10)

        # Create shared legend
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Model prediction'),
            mpatches.Patch(color='green', alpha=0.3, label=r'$\pm 1\sigma$'),
            mpatches.Patch(color='yellow', alpha=0.3, label=r'$\pm 2\sigma$'),
            mpatches.Patch(color='coral', alpha=0.2, label=r'$\pm 3\sigma$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=7, label='Data Points')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), fontsize='medium', frameon=False, bbox_to_anchor=(0.5, 1.05))

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()

        return fig