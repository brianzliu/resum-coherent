import numpy as np
import copy
import GPy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DeepGPModel():
    def __init__(self, trainings_data, normalize_data=True):
        self.trainings_data = copy.deepcopy(trainings_data)
        self.fidelities = list(self.trainings_data.keys())
        self.nfidelities = len(self.fidelities)
        self.normalize_data = normalize_data
        self.X_scalers = {}
        self.Y_scalers = {}
        self.gp1 = None
        self.gp2 = None

    def set_trainings_data(self, trainings_data):
        self.trainings_data = copy.deepcopy(trainings_data)

    def normalize_xy(self, x_list, y_list):
        x_list_norm = []
        y_list_norm = []

        for idx, (x, y) in enumerate(zip(x_list, y_list)):
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            x_norm = scaler_x.fit_transform(x)
            y_norm = scaler_y.fit_transform(y)
            self.X_scalers[self.fidelities[idx]] = scaler_x
            self.Y_scalers[self.fidelities[idx]] = scaler_y

            x_list_norm.append(x_norm)
            y_list_norm.append(y_norm)

        return x_list_norm, y_list_norm

    def build_model(self, n_inducing=30, n_latent_dims=None, n_restarts=10, noise_dict=None):
        """
        Build a Deep GP manually.
        """
        x_train = []
        y_train = []
        
        for fidelity in self.fidelities:
            x_tmp = np.atleast_2d(self.trainings_data[fidelity][0])
            y_tmp = np.atleast_2d(self.trainings_data[fidelity][1]).T
            fidelity_label = np.ones((x_tmp.shape[0], 1)) * fidelity
            x_tmp = np.hstack([x_tmp, fidelity_label])

            x_train.append(x_tmp)
            y_train.append(y_tmp)
        
        X_train = np.vstack(x_train)
        Y_train = np.vstack(y_train)

        if self.normalize_data:
            X_train, Y_train = self.normalize_xy([X_train], [Y_train])
            X_train, Y_train = X_train[0], Y_train[0]

        input_dim = X_train.shape[1]

        if n_latent_dims is None:
            if input_dim <= 2:
                n_latent_dims = 2
            elif input_dim <= 5:
                n_latent_dims = 5
            elif input_dim <= 10:
                n_latent_dims = 6
            else:
                n_latent_dims = min(7, input_dim)

        print(f"Using latent dimension: {n_latent_dims}")

        # Build GP1: input -> latent
        kernel1 = (GPy.kern.RBF(input_dim, ARD=True) +
                   GPy.kern.Bias(input_dim) +
                   GPy.kern.Matern32(input_dim, ARD=True))

        self.gp1 = GPy.models.SparseGPRegression(
            X_train, 
            np.random.randn(X_train.shape[0], n_latent_dims), 
            kernel=kernel1,
            num_inducing=n_inducing
        )
        self.gp1.Z[:] = X_train[np.random.choice(X_train.shape[0], n_inducing, replace=False)]

        if noise_dict is not None:
            lf_noise = noise_dict.get(0, 1e-6)
            self.gp1.Gaussian_noise.variance.constrain_bounded(1e-6, 1e-2)  # learnable but small
            print(f"GP1 noise constrained between 1e-6 and 1e-2 (started around {lf_noise})")

        # Build GP2: latent -> output
        kernel2 = GPy.kern.RBF(n_latent_dims, ARD=True) + GPy.kern.Bias(n_latent_dims)

        self.gp2 = GPy.models.SparseGPRegression(
            np.random.randn(X_train.shape[0], n_latent_dims),
            Y_train,
            kernel=kernel2,
            num_inducing=n_inducing
        )
        self.gp2.Z[:] = np.random.randn(n_inducing, n_latent_dims)

        if noise_dict is not None:
            hf_noise = noise_dict.get(1, 1e-8)
            self.gp2.Gaussian_noise.variance.fix(hf_noise)
            print(f"GP2 noise fixed to {hf_noise}")

        # Optimize GP1
        print("Optimizing first GP (input -> latent)...")
        self.gp1.optimize(messages=True, max_iters=500)
        for _ in range(n_restarts):
            self.gp1.optimize(messages=True, max_iters=500)

        # Predict latent variables
        print("Predicting latent variables...")
        latent_mean, _ = self.gp1.predict(X_train)

        # Set input for GP2 training
        self.gp2.set_XY(latent_mean, Y_train)

        # Optimize GP2
        print("Optimizing second GP (latent -> output)...")
        self.gp2.optimize(messages=True, max_iters=500)
        for _ in range(n_restarts):
            self.gp2.optimize(messages=True, max_iters=500)

        print("Deep GP model built and optimized.")
        return self

    def predict(self, test_data):
        """
        Predict outputs for given test data at all fidelities.

        test_data: dict of {fidelity: (X_test, Y_test)}
        
        Returns:
            dict of {fidelity: (Y_pred, Y_var)}
        """
        predictions = {}

        for fidelity, (X_test, _) in test_data.items():
            X_test = np.asarray(X_test)

            # Add fidelity index
            fidelity_column = np.ones((X_test.shape[0], 1)) * fidelity
            X_test_augmented = np.hstack([X_test, fidelity_column])

            if self.normalize_data:
                scaler_x = self.X_scalers.get(self.fidelities[0])
                X_test_augmented = scaler_x.transform(X_test_augmented)

            # Step 1: Input -> Latent
            latent_pred, _ = self.gp1.predict(X_test_augmented)

            # Step 2: Latent -> Output
            Y_pred, Y_var = self.gp2.predict(latent_pred)

            if self.normalize_data:
                scaler_y = self.Y_scalers.get(self.fidelities[0])
                Y_pred = scaler_y.inverse_transform(Y_pred)

            predictions[fidelity] = (Y_pred, Y_var)

        return predictions
    
import numpy as np
import GPy
import copy
import paramz
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches



class DeepGPModel():
    def __init__(self, trainings_data, normalize_data=True):
        self.trainings_data = copy.deepcopy(trainings_data)
        self.fidelities = list(self.trainings_data.keys())
        self.nfidelities = len(self.fidelities)
        self.normalize_data = normalize_data
        self.X_scalers = {}
        self.Y_scalers = {}
        self.deepgp_lf = None  # Deep GP for low fidelity
        self.gp_delta = None   # GP for correction
        self.rho = 0.01        # Initial scaling

    def set_trainings_data(self, trainings_data):
        self.trainings_data = copy.deepcopy(trainings_data)

    def normalize_xy(self, x_list, y_list):
        x_list_norm = []
        y_list_norm = []

        for idx, (x, y) in enumerate(zip(x_list, y_list)):
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            x_norm = scaler_x.fit_transform(x)
            y_norm = scaler_y.fit_transform(y)
            self.X_scalers[self.fidelities[idx]] = scaler_x
            self.Y_scalers[self.fidelities[idx]] = scaler_y

            x_list_norm.append(x_norm)
            y_list_norm.append(y_norm)

        return x_list_norm, y_list_norm

    def build_model(self, n_inducing=30, n_latent_dims=None, n_restarts=10, noise_dict=None):
        """
        Build Deep GP for LF and GP for correction.
        rho is optimized separately after training.
        """
        assert 0 in self.trainings_data and 1 in self.trainings_data, "Trainings data must contain fidelity 0 (LF) and 1 (HF)."

        X_lf, Y_lf = self.trainings_data[0]
        X_hf, Y_hf = self.trainings_data[1]

        X_lf = np.atleast_2d(X_lf)
        Y_lf = np.atleast_2d(Y_lf).T
        X_hf = np.atleast_2d(X_hf)
        Y_hf = np.atleast_2d(Y_hf).T

        if self.normalize_data:
            [X_lf, X_hf], [Y_lf, Y_hf] = self.normalize_xy([X_lf, X_hf], [Y_lf, Y_hf])

        input_dim = X_lf.shape[1]

        if n_latent_dims is None:
            if input_dim <= 2:
                n_latent_dims = 2
            elif input_dim <= 5:
                n_latent_dims = 5
            elif input_dim <= 10:
                n_latent_dims = 6
            else:
                n_latent_dims = min(7, input_dim)

        print(f"Using latent dimension for LF Deep GP: {n_latent_dims}")

        # --- Deep GP (actually simple Sparse GP) ---
        kernel1 = (GPy.kern.RBF(input_dim, ARD=True) +
                   GPy.kern.Bias(input_dim) +
                   GPy.kern.Matern32(input_dim, ARD=True))

        self.deepgp_lf = GPy.models.SparseGPRegression(
            X_lf,
            np.random.randn(X_lf.shape[0], n_latent_dims),
            kernel=kernel1,
            num_inducing=n_inducing
        )
        self.deepgp_lf.Z[:] = X_lf[np.random.choice(X_lf.shape[0], n_inducing, replace=False)]

        if noise_dict is not None:
            lf_noise = noise_dict[0]
            #self.deepgp_lf.Gaussian_noise.variance.constrain_bounded(1e-6, 1e-2)
            self.deepgp_lf.Gaussian_noise.variance.fix(lf_noise)
            print(f"Deep GP LF noise fixed to {lf_noise}")

        # --- Optimize Deep GP ---
        print("Optimizing Deep GP for LF...")
        self.deepgp_lf.optimize(messages=True, max_iters=500)
        for _ in range(n_restarts):
            self.deepgp_lf.optimize(messages=True, max_iters=500)

        # --- Correction GP ---
        print("Training correction GP...")
        lf_pred_hf, _ = self.deepgp_lf.predict(X_hf)
        residuals = Y_hf - self.rho * lf_pred_hf  # initial guess with rho=0.01

        kernel_delta = GPy.kern.RBF(input_dim, ARD=True)
        self.gp_delta = GPy.models.GPRegression(X_hf, residuals, kernel_delta)

        if noise_dict is not None:
            #hf_noise = noise_dict.get(1, 1e-8)
            self.gp_delta.Gaussian_noise.variance.fix(lf_noise)
            #print(f"Correction GP noise fixed to {hf_noise}")

        self.gp_delta.optimize(messages=True, max_iters=500)
        for _ in range(n_restarts):
            self.gp_delta.optimize(messages=True, max_iters=500)

        # --- Optimize rho ---
        print("Optimizing rho...")

        def mse_rho(rho_array):
            rho_val = rho_array[0]
            pred_lf, _ = self.deepgp_lf.predict(X_hf)
            pred_delta, _ = self.gp_delta.predict(X_hf)
            pred_total = rho_val * pred_lf + pred_delta
            return np.mean((pred_total - Y_hf)**2)

        res = minimize(mse_rho, x0=[self.rho], bounds=[(1e-5, 1.0)])
        self.rho = res.x[0]

        print(f"Learned rho: {self.rho:.5f}")
        print("Deep Multi-Fidelity Correction Model built and optimized.")

        # --- Empirical covariance estimation between LF and correction at HF points ---
        lf_pred_train, _ = self.deepgp_lf.predict(X_hf)
        delta_pred_train, _ = self.gp_delta.predict(X_hf)

        self.empirical_covariance = np.cov(lf_pred_train.squeeze(), delta_pred_train.squeeze())[0, 1]
        print(f"Estimated empirical covariance: {self.empirical_covariance:.5e}")
        

        return self

    def predict(self, test_data):
        predictions = {}

        for fidelity, (X_test, _) in test_data.items():
            X_test = np.atleast_2d(X_test)

            if self.normalize_data:
                scaler_x = self.X_scalers.get(fidelity, self.X_scalers[0])
                X_test_norm = scaler_x.transform(X_test)
            else:
                X_test_norm = X_test

            # Predict using Deep GP (low fidelity model)
            lf_pred, lf_var = self.deepgp_lf.predict(X_test_norm)

            """
            if fidelity == 0:
                # --- Predicting Low Fidelity ---
                y_pred = lf_pred
                y_var = lf_var
            else:
                # --- Predicting High Fidelity ---
                delta_pred, delta_var = self.gp_delta.predict(X_test_norm)
                y_pred = self.rho * lf_pred + delta_pred
                y_var = (self.rho ** 2) * lf_var + delta_var
            """
            
            if fidelity == 0:
                # Predicting LF
                y_pred = lf_pred
                y_var = lf_var
            else:
                # Predicting HF
                delta_pred, delta_var = self.gp_delta.predict(X_test_norm)
                y_pred = self.rho * lf_pred + delta_pred
                y_var = (self.rho ** 2) * lf_var + delta_var + 2 * self.rho * self.empirical_covariance

            # Undo normalization if needed
            if self.normalize_data:
                scaler_y = self.Y_scalers.get(fidelity, self.Y_scalers[0])
                y_pred = scaler_y.inverse_transform(y_pred)
                y_var = y_var * (scaler_y.scale_[0] ** 2)  # Variance rescales as (scale_factor)^2

            predictions[fidelity] = (y_pred, y_var)

        return predictions


    def get_marginalized_predictions(self, grid_steps=40, marginal_steps=30, random_seed=42):
        """
        Computes marginalized predictions for each feature separately by averaging predictions
        over all other dimensions.
        
        Parameters:
        - grid_steps: Number of grid points along the selected feature.
        - marginal_steps: Number of random samples for marginalization over the other dimensions.
        - random_seed: Random seed for reproducibility.
        """
        np.random.seed(random_seed)
        self.predictions_marginalized = {}
        fidelities = list(self.trainings_data.keys())

        for f in fidelities:
            pred_info = {'x': [], 'mean': [], 'std': []}

            X_train, _ = self.trainings_data[f]
            if isinstance(X_train, list):
                X_train = np.asarray(X_train)

            input_dim = X_train.shape[1]

            # Get unnormalized ranges
            if self.normalize_data:
                scaler_x = self.X_scalers[f]
                X_train = scaler_x.inverse_transform(X_train)

            x_mins = np.min(X_train, axis=0)
            x_maxs = np.max(X_train, axis=0)

            for dim in range(input_dim):
                # Set up grid for the dimension of interest
                x_linspace = np.linspace(x_mins[dim], x_maxs[dim], grid_steps)
                marginalized_means = []
                marginalized_stds = []

                for val in x_linspace:
                    # Sample other dimensions uniformly
                    X_query = np.random.uniform(x_mins, x_maxs, size=(marginal_steps, input_dim))
                    X_query[:, dim] = val  # Fix the dimension of interest

                    # Normalize if needed
                    if self.normalize_data:
                        scaler_x = self.X_scalers[f]
                        X_query = scaler_x.transform(X_query)

                    # Predict
                    X_query_dict = {f: (X_query, np.zeros((marginal_steps, 1)))}
                    preds = self.predict(X_query_dict)
                    y_pred, y_var = preds[f]
                    y_pred = y_pred.flatten()

                    # Marginalize: take mean and std
                    marginalized_means.append(np.mean(y_pred))
                    marginalized_stds.append(np.std(y_pred))

                pred_info['x'].append(x_linspace)
                pred_info['mean'].append(np.array(marginalized_means))
                pred_info['std'].append(np.array(marginalized_stds))

            self.predictions_marginalized[f] = pred_info

        return self

    def plot_marginalized_predictions(self, x_data=None, y_data=None):
        """
        Plots the marginalized predictions (mean ± uncertainty) for each feature.
        """
        if not hasattr(self, 'predictions_marginalized'):
            raise ValueError("Marginalized predictions not found. Run get_marginalized_predictions() first.")

        n_fidelities = len(self.predictions_marginalized)
        n_features = len(next(iter(self.predictions_marginalized.values()))['x'])

        fig, axes = plt.subplots(nrows=n_fidelities, ncols=n_features, figsize=(5 * n_features, 4 * n_fidelities), squeeze=False)

        for f_idx, (fidelity, pred_fid) in enumerate(self.predictions_marginalized.items()):
            for dim in range(n_features):
                ax = axes[f_idx][dim]

                x = pred_fid['x'][dim]
                y_mean = pred_fid['mean'][dim]
                y_std = pred_fid['std'][dim]

                if len(x) != len(y_mean):
                    print(f"Skipping dim {dim} in fidelity {fidelity}: x and y_mean lengths do not match.")
                    continue

                # Plot uncertainty bands
                ax.fill_between(x, y_mean - 3*y_std, y_mean + 3*y_std, color="coral", alpha=0.2, label=r'$\pm 3\sigma$')
                ax.fill_between(x, y_mean - 2*y_std, y_mean + 2*y_std, color="yellow", alpha=0.3, label=r'$\pm 2\sigma$')
                ax.fill_between(x, y_mean - 1*y_std, y_mean + 1*y_std, color="green", alpha=0.3, label=r'$\pm 1\sigma$')

                # Plot mean prediction
                ax.plot(x, y_mean, color="black", linewidth=2, label="Prediction Mean")

                # Optionally plot training data
                if x_data is not None and y_data is not None:
                    if fidelity in x_data and fidelity in y_data:
                        Xd, Yd = x_data[fidelity]
                        if isinstance(Xd, list):
                            Xd = np.asarray(Xd)
                        if isinstance(Yd, list):
                            Yd = np.asarray(Yd)
                        if Xd.shape[0] == Yd.shape[0]:
                            ax.scatter(Xd[:, dim], Yd.flatten(), color="blue", s=15, alpha=0.7, label="Data")

                ax.set_xlabel(f"Feature {dim}", fontsize=12)
                ax.set_ylabel(f"Predicted Y", fontsize=12)
                ax.set_title(f"Fidelity {fidelity} - Feature {dim}", fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.tick_params(axis='both', which='major', labelsize=10)

        # Create shared legend
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Prediction Mean'),
            mpatches.Patch(color='green', alpha=0.3, label=r'$\pm 1\sigma$'),
            mpatches.Patch(color='yellow', alpha=0.3, label=r'$\pm 2\sigma$'),
            mpatches.Patch(color='coral', alpha=0.2, label=r'$\pm 3\sigma$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=7, label='Data Points')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), fontsize='medium', frameon=False, bbox_to_anchor=(0.5, 1.02))

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()

        return fig

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
        self.rho = 0.01  # Initial scaling

    def set_trainings_data(self, trainings_data):
        self.trainings_data = copy.deepcopy(trainings_data)

    def normalize_xy(self, x_list, y_list):
        x_list_norm = []
        y_list_norm = []

        for idx, (x, y) in enumerate(zip(x_list, y_list)):
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            x_norm = scaler_x.fit_transform(x)
            y_norm = scaler_y.fit_transform(y)
            self.X_scalers[self.fidelities[idx]] = scaler_x
            self.Y_scalers[self.fidelities[idx]] = scaler_y

            x_list_norm.append(x_norm)
            y_list_norm.append(y_norm)

        return x_list_norm, y_list_norm

    def build_model(self, n_inducing=30, n_latent_dims=2, n_restarts=5, noise_dict=None):
        assert 0 in self.trainings_data and 1 in self.trainings_data, "Trainings data must contain fidelity 0 (LF) and 1 (HF)."

        X_lf, Y_lf = self.trainings_data[0]
        X_hf, Y_hf = self.trainings_data[1]

        X_lf = np.atleast_2d(X_lf)
        Y_lf = np.atleast_2d(Y_lf).T
        X_hf = np.atleast_2d(X_hf)
        Y_hf = np.atleast_2d(Y_hf).T

        if self.normalize_data:
            [X_lf, X_hf], [Y_lf, Y_hf] = self.normalize_xy([X_lf, X_hf], [Y_lf, Y_hf])

        input_dim = X_lf.shape[1]

        print(f"Building Deep GP with latent dimension {n_latent_dims}")

        # Initialize latent variables
        N = X_lf.shape[0]
        H_init = np.random.randn(N, n_latent_dims)

        # GP1: X -> H
        kernel1 = GPy.kern.RBF(input_dim, ARD=True)
        self.gp1_models = []
        for d in range(n_latent_dims):
            model = GPy.models.SparseGPRegression(X_lf, H_init[:, d:d+1], kernel=kernel1.copy(), num_inducing=n_inducing)
            if noise_dict is not None:
                model.Gaussian_noise.variance.fix(noise_dict.get(0, 0.02))
            self.gp1_models.append(model)

        # GP2: H -> Y
        kernel2 = GPy.kern.RBF(n_latent_dims, ARD=True)
        self.gp2_model = GPy.models.SparseGPRegression(H_init, Y_lf, kernel=kernel2, num_inducing=n_inducing)
        if noise_dict is not None:
            self.gp2_model.Gaussian_noise.variance.fix(noise_dict.get(0, 0.02))

        # Optimize Deep GP
        print("Optimizing Deep GP for LF...")
        for _ in range(n_restarts):
            for model in self.gp1_models:
                model.optimize(messages=False, max_iters=500)
            self.gp2_model.optimize(messages=False, max_iters=500)

        # Build Correction GP for HF
        print("Building correction GP...")
        lf_pred_hf, _ = self.predict_lf(X_hf)
        residuals = Y_hf - self.rho * lf_pred_hf

        kernel_delta = GPy.kern.RBF(input_dim, ARD=True)
        self.gp_delta = GPy.models.GPRegression(X_hf, residuals, kernel_delta)
        if noise_dict is not None:
            hf_noise = noise_dict.get(1, 1e-8)
            self.gp_delta.Gaussian_noise.variance = hf_noise  # Initialize it small
            self.gp_delta.Gaussian_noise.variance.constrain_bounded(1e-8, 1e-5)
        else:
            self.gp_delta.Gaussian_noise.variance.constrain_bounded(1e-8, 1e-5)

        self.gp_delta.optimize(messages=True, max_iters=500)
        for _ in range(n_restarts):
            self.gp_delta.optimize(messages=True, max_iters=500)

        # Optimize rho
        print("Optimizing rho...")
        def mse_rho(rho_array):
            rho_val = rho_array[0]
            pred_lf, _ = self.predict_lf(X_hf)
            pred_delta, _ = self.gp_delta.predict(X_hf)
            pred_total = rho_val * pred_lf + pred_delta
            return np.mean((pred_total - Y_hf)**2)

        res = minimize(mse_rho, x0=[self.rho], bounds=[(1e-5, 1.0)])
        self.rho = res.x[0]

        # Estimate empirical covariance
        lf_pred_train, _ = self.predict_lf(X_hf)
        delta_pred_train, _ = self.gp_delta.predict(X_hf)
        self.empirical_covariance = np.cov(lf_pred_train.squeeze(), delta_pred_train.squeeze())[0, 1]
        print(f"Learned rho: {self.rho:.5f}")
        print("Deep Multi-Fidelity Correction Model built and optimized.")

        return self

    def predict_lf(self, X_test):
        H_pred_list = []
        for model in self.gp1_models:
            mu, _ = model.predict(X_test)
            H_pred_list.append(mu)

        H_pred = np.hstack(H_pred_list)
        Y_pred, Y_var = self.gp2_model.predict(H_pred)
        return Y_pred, Y_var

    def predict(self, test_data):
        predictions = {}

        for fidelity, (X_test, _) in test_data.items():
            X_test = np.atleast_2d(X_test)

            if self.normalize_data:
                scaler_x = self.X_scalers.get(fidelity, self.X_scalers[0])
                X_test_norm = scaler_x.transform(X_test)
            else:
                X_test_norm = X_test

            # Predict LF output
            lf_pred, lf_var = self.predict_lf(X_test_norm)

            if fidelity == 0:
                y_pred = lf_pred
                y_var = lf_var
            else:
                delta_pred, delta_var = self.gp_delta.predict(X_test_norm)
                y_pred = self.rho * lf_pred + delta_pred
                y_var = (self.rho ** 2) * lf_var + delta_var + 2 * self.rho * self.empirical_covariance

            if self.normalize_data:
                scaler_y = self.Y_scalers.get(fidelity, self.Y_scalers[0])
                y_pred = scaler_y.inverse_transform(y_pred)
                y_var = y_var * (scaler_y.scale_[0] ** 2)

            predictions[fidelity] = (y_pred, y_var)

        return predictions