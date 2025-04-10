import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTENSOR_FLAGS"] = "compiledir=./pytensor_cache,mode=FAST_COMPILE,optimizer=None"
print("Compiledir:", os.environ.get("PYTENSOR_FLAGS"))
import pymc as pm
import logging
logging.getLogger("pymc").setLevel(logging.ERROR)
#logging.getLogger("pytensor").setLevel(logging.ERROR)
from itertools import combinations_with_replacement
from numpy.polynomial.legendre import Legendre
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from itertools import product, combinations
import random

# Set seeds for reproducibility
np.random.seed(42)         # NumPy seed
random.seed(42)            # Python random seed


class PCEMultiFidelityModel:
    def __init__(self, trainings_data, priors, parameters, degree=None):
        """
        Initialize the multi-fidelity model.

        Parameters:
        - basis_matrices (dict): Dictionary of basis matrices for each fidelity level.
          Example: {"lf": basis_matrix_lf, "mf": basis_matrix_mf, "hf": basis_matrix_hf}
        - trainings_data (dict): Dictionary of observed data for each fidelity level.
          Example: {"lf": [x_lf, y_lf], "mf": [x_mf,y_mf], "hf": [x_hf,y_hf]}
        - indices (dict): Dictionary of indices mapping one fidelity level to the next.
          Example: {"mf": indices_mf, "hf": indices_hf}
        - priors (dict): Dictionary of prior configurations for each fidelity level.
          Example: {"lf": {"sigma": 0.5}, "mf": {"sigma": 0.1}, "hf": {"sigma": 0.01}}
        """
        
        self.trainings_data = trainings_data
        self.fidelities = list(self.trainings_data.keys())
        self.feature_labels = list(map(str, parameters.keys()))
        
        self.x_min = np.array([parameters[k][0] for k in self.feature_labels])
        self.x_max = np.array([parameters[k][1] for k in self.feature_labels])
        if np.any(self.x_min < -1.) or np.any(self.x_max > 1.):
                print("Data outside [-1,1] detected. Rescaling features...")
                for f in self.fidelities:
                    x_data = self.trainings_data[f][0]
                    self.trainings_data[f][0] = self.normalize_to_minus1_plus1(x_data)

        self.priors = priors
        self.degree = degree
        if self.degree==None:
            self.degree = self.find_optimal_order()
        self.model = None
        self.trace = None

        self.basis_matrices = {}
        self.indices = {}
        for f in self.fidelities:
            x_data = trainings_data[f][0]
            self.basis_matrices[f] = self._generate_basis(x_data)
        
        for i,f in enumerate(self.fidelities[1:]):
            self.indices[f] = self.find_indices(trainings_data[f][0],trainings_data[self.fidelities[i]][0])

    def normalize_to_minus1_plus1(self,x):
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
    

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
    
    @staticmethod
    def find_indices(x_hf, x_lf):
        """
        Finds the indices of x_hf in x_lf.

        Parameters:
        - x_hf (numpy.ndarray): Array of high-fidelity x values.
        - x_lf (numpy.ndarray): Array of low-fidelity x values.

        Returns:
        - list: Indices of x_hf in x_lf.
        """
        indices = []
        for x in x_hf:
            idx = np.where((x_lf == x).all(axis=1))[0]
            if idx.size > 0:
                indices.append(idx[0])  # Append the index
            else:
                raise ValueError(f"Value {x} from x_hf not found in x_lf.")
        return indices
    
    @staticmethod
    def multivariate_legendre_with_interactions(order, x):
        """
        Generate multivariate Legendre polynomial basis with interaction terms.
        
        Parameters:
        - order (int): Maximum polynomial degree.
        - x (ndarray): Input data of shape (n_samples, n_features).
        
        Returns:
        - basis (ndarray): Basis matrix including interactions.
        """

        n_samples, n_features = x.shape
        degrees = list(product(range(order + 1), repeat=n_features))
        basis = []
        for degree in degrees:
            term = np.ones(n_samples)
            for i, d in enumerate(degree):
                term *= np.polynomial.legendre.Legendre.basis(d)(x[:, i])
            basis.append(term)

        # Add interaction terms
        for i, j in combinations(range(n_features), 2):
            basis.append(x[:, i] * x[:, j])

        return np.vstack(basis).T

    def _add_fidelity(self, model, fidelity, y_prev_pred_full):
        """
        Recursively add fidelity levels to the model.

        Parameters:
        - model (pm.Model): The PyMC model.
        - fidelity_chain (list): List of fidelities to be added (e.g., ["lf", "mf", "hf"]).
        - prev_pred (pm.Deterministic): The prediction from the previous fidelity level.

        Returns:
        - pm.Deterministic: Final prediction for the highest fidelity level.
        """

        # Basis matrix and observed data
        basis_matrix = self.basis_matrices[fidelity]
        observed = self.trainings_data[fidelity][1]

        #y_prev_pred_subset = pm.Deterministic(f"y_prev_pred_subset_{fidelity}", y_prev_pred_full[self.indices[fidelity]])
        # fix 
        subset_indices = self.indices[fidelity]
        y_prev_pred_subset = pm.Deterministic(
            f"y_prev_pred_subset_{fidelity}",
            pm.math.stack([y_prev_pred_full[i] for i in subset_indices])
        )
        # Scaling factor
        rho = pm.Normal(f"rho_{fidelity}", mu=1, sigma=self.priors[fidelity]["sigma_rho"])
        # Priors for high-fidelity discrepancy coefficients
        coeffs_delta = pm.Normal(f"coeffs_delta_{fidelity}", mu=0, sigma=self.priors[fidelity]["sigma_coeffs_delta"], shape=self.basis_matrices[fidelity].shape[1])
        # High-fidelity discrepancy
        #delta_pred = pm.Deterministic(f"delta_{fidelity}", pm.math.dot(self.basis_matrices[fidelity], coeffs_delta))
        # fix
        basis = pm.Data(f"basis_{fidelity}_delta", self.basis_matrices[fidelity])
        delta_pred = pm.Deterministic(f"delta_{fidelity}", pm.math.dot(basis, coeffs_delta))

            
        # High-fidelity predictions
        y_pred = pm.Deterministic(f"y_pred_{fidelity}", rho * y_prev_pred_subset + delta_pred)
        # Likelihood for high-fidelity data
        sigma = pm.HalfNormal(f"sigma_{fidelity}", sigma=self.priors[fidelity]["sigma"])
        y_likeli = pm.Normal(f"y_likeli_{fidelity}", mu=y_pred, sigma=sigma, observed=self.trainings_data[fidelity][1])
        obs = self.trainings_data[fidelity][1]
        # Compute the pointwise log likelihood manually using pm.math functions.
        # For a Normal distribution, the log likelihood is:
        log_lik = -0.5 * (((obs - y_pred) / sigma) ** 2) - pm.math.log(sigma) - 0.5 * pm.math.log(2 * np.pi)
        # Register it as a Deterministic node so it is tracked in the InferenceData.
        pm.Deterministic(f"log_likelihood_{fidelity}", log_lik)

        return y_pred

    def build_model(self):
        """
        Build the PyMC multi-fidelity model recursively.
        """
          # ["lf", "mf", "hf"]
        with pm.Model() as model:
            # Start with low-fidelity coefficients
            coeffs = pm.Normal(f"coeffs_{self.fidelities[0]}",
                mu=0,
                sigma=self.priors[self.fidelities[0]]["sigma_coeffs"],
                shape=self.basis_matrices[self.fidelities[0]].shape[1]
            )

            #y_prev_pred_full = pm.Deterministic(f"y_pred_full_{self.fidelities[0]}", pm.math.dot(self.basis_matrices[self.fidelities[0]], coeffs))
            #fixed
            basis = pm.Data(f"basis_{self.fidelities[0]}", self.basis_matrices[self.fidelities[0]])
            y_prev_pred_full = pm.Deterministic(f"y_pred_full_{self.fidelities[0]}", pm.math.dot(basis, coeffs))
           
            sigma = pm.HalfNormal(f"sigma_{self.fidelities[0]}", sigma=self.priors[self.fidelities[0]]["sigma"])
            y_likeli = pm.Normal(f"y_likeli_{self.fidelities[0]}", mu=y_prev_pred_full, sigma=sigma, observed=self.trainings_data[self.fidelities[0]][1])
            obs = self.trainings_data[self.fidelities[0]][1]
            # Compute the pointwise log likelihood manually using pm.math functions.
            # For a Normal distribution, the log likelihood is:
            log_lik = -0.5 * (((obs - y_prev_pred_full) / sigma) ** 2) - pm.math.log(sigma) - 0.5 * pm.math.log(2 * np.pi)
            # Register it as a Deterministic node so it is tracked in the InferenceData.
            pm.Deterministic(f"log_likelihood_{self.fidelities[0]}", log_lik)
            
            # Add fidelities recursively
            for fidelity in self.fidelities[1:]:
                y_prev_pred_full = self._add_fidelity(model, fidelity, y_prev_pred_full)
            self.model = model
            
            #pm.model_to_graphviz(model)

            self.model = model
    
    def run_inference(self, method="nuts", n_samples=200, n_steps=1000, tune=100, chains=4, cores=1):
        """
        Run inference on the PCE model.

        Parameters:
        - method (str): Inference method ("advi" or "nuts").

        Returns:
        - pm.backends.base.MultiTrace: The posterior samples.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        with self.model:
            #init_point = self.model.initial_point()
            #print(init_point)
            if method == "advi":
                # Variational Inference
                #approx = pm.fit(n=n_steps, method=pm.MeanField(), progressbar=True)
                #approx = pm.fit(n=n_steps, method=pm.MeanField(), obj_optimizer=pm.adam(learning_rate=1e-2), progressbar=True)
                approx = pm.fit(n=n_steps, method="advi", progressbar=True)
                #approx = pm.fit(n=n_steps, method="advi", progressbar=True, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-2)])
                #plt.plot(approx.hist)
                self.trace = approx.sample(n_samples)
            elif method == "nuts":
                # HMC Sampling
                self.trace = pm.sample(n_samples, tune=tune, chains=chains,init="adapt_diag", target_accept=0.99, cores=cores, progressbar=True)
            else:
                raise ValueError(f"Unknown inference method: {method}")

        return self.trace
    
    def plot_trace(self):
        az.plot_trace(self.trace)
        plt.tight_layout()
        plt.show()
    
    def plot_energy(self):
        az.plot_energy(self.trace)

    def plot_forrest(self):
        param_names = [rv.name for rv in self.model.free_RVs]
        az.plot_forest(self.trace, var_names=param_names)
    
    def save_trace(self, output, version="v1.0"):
        az.to_netcdf(self.trace, f"{output}/pce_{version}_trace.nc")

    def sanity_check_of_basis(self):
        for f in self.fidelities:
            print("X range:", np.min(self.trainings_data[f][0], axis=0), np.max(self.trainings_data[f][0], axis=0))
            print("y_lf std:", np.std(self.trainings_data[f][1]))
            print("Basis stds:", np.std(self.basis_matrices[f], axis=0))
            plt.figure()
            plt.imshow(self.basis_matrices[f], aspect='auto', cmap='magma')
            plt.colorbar()
            plt.title(f"{f} Basis Matrix")
            plt.show()  

    def check_logp_per_variable(self):
        with self.model:
            init_point = self.model.initial_point()
            for rv in self.model.free_RVs:
                logp_fn = self.model.compile_logp(vars=[rv])
                try:
                    logp_val = logp_fn({rv.name: init_point[rv.name]})
                    print(f"{rv.name:25} logp: {logp_val}")
                except Exception as e:
                    print(f"{rv.name:25} logp: ERROR → {e}")

    def find_optimal_order(self):
        """
        Find the optimal polynomial order using cross-validation.

        Parameters:
        - max_order (int): Maximum polynomial order to test.
        - n_splits (int): Number of splits for cross-validation.

        Returns:
        - optimal_order (int): Optimal polynomial order.
        """
        print("Finding the optimal polynomial order using cross-validation...")
        n_splits,max_order = self.trainings_data[self.fidelities[0]][0].shape

        for f in self.fidelities[1:]:
            n_splits_tmp,_ = self.trainings_data[f][0].shape
            if n_splits_tmp < n_splits:
                n_splits = n_splits_tmp

        errors = []
        kf = KFold(n_splits=n_splits)

        for order in range(1, max_order + 1):
            # Generate basis for LF and HF
            basis_with_interactions = {}
            c = {}
            for fidelity in self.fidelities:
                basis_with_interactions[fidelity]=self.multivariate_legendre_with_interactions(order, self.trainings_data[fidelity][0])
                c[fidelity] = np.linalg.lstsq(basis_with_interactions[fidelity],self.trainings_data[fidelity][1], rcond=None)[0]
                # Predict LF contributions to HF
            y_pred = {}
            for i,fidelity in enumerate(self.fidelities[1:]):
                y_pred[f"{self.fidelities[i]}_{fidelity}"] = basis_with_interactions[fidelity] @ c[self.fidelities[i]]
                delta = self.trainings_data[fidelity][1] - y_pred[f"{self.fidelities[i]}_{fidelity}"]

                mse_fold = []
                x_train = {}; y_train = {}
                x_test  = {}; y_test = {}

                for train_idx, test_idx in kf.split(self.trainings_data[fidelity][0]):
                    x_train[fidelity], x_test[fidelity] = self.trainings_data[fidelity][0][train_idx], self.trainings_data[fidelity][0][test_idx]
                    y_train[fidelity], y_test[fidelity] = self.trainings_data[fidelity][1][train_idx], self.trainings_data[fidelity][1][test_idx]

                    # Generate basis matrices for train and test
                    phi_train = self.multivariate_legendre_with_interactions(order, x_train[fidelity])
                    phi_test = self.multivariate_legendre_with_interactions(order, x_test[fidelity])

                    # Fit HF correction
                    c_hf = np.linalg.lstsq(phi_train, y_train[fidelity]-phi_train @ c[self.fidelities[i-1]], rcond=None)[0]
                    y_pred_fold = phi_test @ c_hf + phi_test @ c[self.fidelities[i-1]]

                    mse_fold.append(mean_squared_error(y_test[fidelity], y_pred_fold))

                # Append mean error for this order
                errors.append(np.mean(mse_fold))

        # Return the optimal order (1-based indexing for order)
        optimal_order = np.argmin(errors) + 1
        print(f"The optimal order is {optimal_order}")
        return optimal_order

    def add_log_likelihood_manually(self):
        if "log_likelihood" not in self.trace.groups():
            ll = self.trace.posterior["log_likelihood_lf"]
        print(self.trace)
        ll = self.trace.posterior["log_likelihood_lf"]
        print(self.trace)
