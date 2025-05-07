import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import arviz as az
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTENSOR_FLAGS"] = "compiledir=./pytensor_cache,mode=FAST_COMPILE,optimizer=None"
print("Compiledir:", os.environ.get("PYTENSOR_FLAGS"))
import pymc as pm
from scipy.optimize import minimize
import logging

#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)
#logging.getLogger("pymc").setLevel(logging.ERROR)
#logging.getLogger("pytensor").setLevel(logging.ERROR)
#import warnings
#warnings.filterwarnings("ignore", category=UserWarning, module="arviz")
from itertools import combinations_with_replacement
from numpy.polynomial.legendre import Legendre
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from itertools import product, combinations
import random
from math import comb
import pandas as pd
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
        
        if degree==None:
            order = self.find_optimal_order()
            print("find order: ",order)
            degree = order
        self.degree = [degree for f in range(len(self.fidelities))]
        self.model = None
        self.trace = None
        self.posterior_predictive = None
        self.elbo = None

        self.basis_matrices = {}
        self.indices = {}
        for i,f in enumerate(self.fidelities):
            x_data = trainings_data[f][0]
            self.basis_matrices[f], self.degree[i] = self._generate_basis(x_data, degree)
        
        for i,f in enumerate(self.fidelities[1:]):
            self.indices[f] = self.find_indices(trainings_data[f][0],trainings_data[self.fidelities[i]][0])
        self.set_table()

    def normalize_to_minus1_plus1(self,x):
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
    
    
    def _generate_basis(self, x_data, degree):
        """
        Generate the multivariate Legendre basis for multi-dimensional inputs.

        Parameters:
        - x_data (ndarray): Input data of shape (n_samples, n_dim).

        Returns:
        - basis_matrix (ndarray): Shape (n_samples, n_terms).
        """
        n_samples, n_dim = x_data.shape
        d=degree

        degree_new = degree
        while (n_samples < 2*comb(n_dim + degree_new, degree_new)):
            degree_new = degree_new-1
            if degree_new==1:
                break


        if degree_new != degree:
            print(
                f"Warning: Not enough data samples to fit a polynomial of order {degree}.\n"
                f"  -> Required: at least {2 * comb(n_dim+degree, degree)} samples\n"
                f"  -> Provided: {n_samples} samples\n"
                f"  -> Polynomial order has been reduced to {degree_new}."
            )
        
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
        for i, x in enumerate(x_hf):
            idx = np.where((x_lf == x).all(axis=1))[0]
            if idx.size > 0:
                indices.append((idx[0],i))  # Append the index

        _, hf_indices_list = zip(*indices)
        hf_indices_list = list(hf_indices_list)
        if len(hf_indices_list) != len(x_hf):
            print(f"Warning: {len(x_hf) - len(hf_indices_list)} high-fidelity samples not found in low-fidelity data. Dropping them")
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
        basis_matrix = self.basis_matrices[self.fidelities[fidelity]]
        observed = self.trainings_data[self.fidelities[fidelity]][1]
        obs_mean=np.mean(observed)
        observed_fminus1 = self.trainings_data[self.fidelities[fidelity-1]][1]
        obs_mean_fminus1 = np.mean(observed_fminus1)

        lf_indices_list, hf_indices_list = zip(*self.indices[self.fidelities[fidelity]])
        lf_indices_list = list(lf_indices_list)
        hf_indices_list = list(hf_indices_list)

        try:
            mu_rho = self.priors[self.fidelities[fidelity]]["mu_rho"]
        except KeyError:
            mu_rho = obs_mean/obs_mean_fminus1

        print("Optimizing mu_rho...")
        def mse_rho(rho_array):
            rho_val = rho_array[0]
            return np.mean((observed[hf_indices_list] - rho_val * observed_fminus1[lf_indices_list])**2)

        res = minimize(mse_rho, x0=[mu_rho], bounds=[(1e-5, 10.0)])
        mu_rho = res.x[0]
        print(f"Optimized mu_rho: {mu_rho:.4f}, std {np.std(observed[hf_indices_list] - mu_rho * observed_fminus1[lf_indices_list]):.4f}")
        #self.priors[self.fidelities[fidelity]]["sigma_rho"] = np.std(observed - mu_rho * observed_fminus1[self.indices[self.fidelities[fidelity]]])
        #y_prev_pred_subset = pm.Deterministic(f"y_prev_pred_subset_{self.fidelities[fidelity]}", y_prev_pred_full[self.indices[self.fidelities[fidelity]]])

        subset_indices = lf_indices_list
        y_prev_pred_subset = pm.Deterministic(
            f"y_prev_pred_subset_{self.fidelities[fidelity]}",
            pm.math.stack([y_prev_pred_full[i] for i in subset_indices])
        )
        # Scaling factor
        
        rho = pm.Normal(f"rho_{self.fidelities[fidelity]}", mu=mu_rho, sigma=self.priors[self.fidelities[fidelity]]["sigma_rho"])
        # Priors for high-fidelity discrepancy coefficients
        coeffs_delta = pm.Normal(f"coeffs_delta_{self.fidelities[fidelity]}", mu=0, sigma=self.priors[self.fidelities[fidelity]]["sigma_coeffs_delta"], shape=self.basis_matrices[self.fidelities[fidelity]].shape[1])
        # High-fidelity discrepancy
        #delta_pred = pm.Deterministic(f"delta_{fidelity}", pm.math.dot(self.basis_matrices[fidelity], coeffs_delta))
        # fix
        basis = pm.Data(f"basis_{self.fidelities[fidelity]}_delta", self.basis_matrices[self.fidelities[fidelity]])
        delta_pred = pm.Deterministic(f"delta_{self.fidelities[fidelity]}", pm.math.dot(basis, coeffs_delta))

            
        # High-fidelity predictions
        raw_pred = rho * y_prev_pred_subset + delta_pred
        #softplus = lambda x: pm.math.log(1 + pm.math.exp(x))
        #raw_pred = softplus(raw_pred)
        y_pred = pm.Deterministic(f"y_pred_{self.fidelities[fidelity]}", raw_pred)
        # Likelihood for high-fidelity data
        sigma = pm.HalfNormal(f"sigma_{self.fidelities[fidelity]}", sigma=self.priors[self.fidelities[fidelity]]["sigma_y"])
        y_likeli = pm.Normal(f"y_likeli_{self.fidelities[fidelity]}", mu=y_pred, sigma=sigma, observed=self.trainings_data[self.fidelities[fidelity]][1][hf_indices_list])
        obs = self.trainings_data[self.fidelities[fidelity]][1]
        # Compute the pointwise log likelihood manually using pm.math functions.
        # For a Normal distribution, the log likelihood is:
        log_lik = -0.5 * (((obs[hf_indices_list] - y_pred) / sigma) ** 2) - pm.math.log(sigma) - 0.5 * pm.math.log(2 * np.pi)
        # Register it as a Deterministic node so it is tracked in the InferenceData.
        pm.Deterministic(f"log_likelihood_{self.fidelities[fidelity]}", log_lik)

        return y_pred

    def build_model(self):
        """
        Build the PyMC multi-fidelity model recursively.
        """
          # ["lf", "mf", "hf"]
        with pm.Model() as model:
            # Start with low-fidelity coefficients
            # Bayesian sparce PCE model using automatic relevance determination
            
            basis = pm.Data(f"basis_{self.fidelities[0]}", self.basis_matrices[self.fidelities[0]])
            if  self.priors[self.fidelities[0]]["sigma_coeffs_prior_type"]=="auto" :
                sigma_coeff = pm.HalfNormal(f"tau_{self.fidelities[0]}",sigma=self.priors[self.fidelities[0]]["sigma_coeffs"], shape=self.basis_matrices[self.fidelities[0]].shape[1])
            elif  self.priors[self.fidelities[0]]["sigma_coeffs_prior_type"]=="cauchy" :
                sigma_coeff = pm.HalfCauchy(f"lambda_{self.fidelities[0]}", beta=self.priors[self.fidelities[0]]["sigma_coeffs"], shape=basis.shape[1])
            elif self.priors[self.fidelities[0]]["sigma_coeffs_prior_type"]=="lasso" :
                # Hierarchical shrinkage prior: ARD Bayesian Lasso
                b = pm.HalfNormal(f"b_{self.fidelities[0]}", sigma=self.priors[self.fidelities[0]]["sigma_coeffs"])
                sigma_coeff = pm.Exponential(f"lambda_{self.fidelities[0]}", lam=1.0 / b, shape=basis.shape[1])
            else:
                sigma_coeff = self.priors[self.fidelities[0]]["sigma_coeffs"]

            coeffs = pm.Normal(f"coeffs_{self.fidelities[0]}",
                mu=0,
                sigma= sigma_coeff,
                shape=self.basis_matrices[self.fidelities[0]].shape[1]
            )

            #y_prev_pred_full = pm.Deterministic(f"y_pred_full_{self.fidelities[0]}", pm.math.dot(self.basis_matrices[self.fidelities[0]], coeffs))
            #fixed
            #basis = pm.Data(f"basis_{self.fidelities[0]}", self.basis_matrices[self.fidelities[0]])
            raw_pred = pm.math.dot(basis, coeffs)
            #softplus = lambda x: pm.math.log(1 + pm.math.exp(x))
            #raw_pred = softplus(raw_pred)
            y_prev_pred_full = pm.Deterministic(f"y_pred_{self.fidelities[0]}", raw_pred)
           
            sigma = pm.HalfNormal(f"sigma_{self.fidelities[0]}", sigma=self.priors[self.fidelities[0]]["sigma_y"])
            y_likeli = pm.Normal(f"y_likeli_{self.fidelities[0]}", mu=y_prev_pred_full, sigma=sigma, observed=self.trainings_data[self.fidelities[0]][1])
            obs = self.trainings_data[self.fidelities[0]][1]
            # Compute the pointwise log likelihood manually using pm.math functions.
            # For a Normal distribution, the log likelihood is:
            log_lik = -0.5 * (((obs - y_prev_pred_full) / sigma) ** 2) - pm.math.log(sigma) - 0.5 * pm.math.log(2 * np.pi)
            # Register it as a Deterministic node so it is tracked in the InferenceData.
            pm.Deterministic(f"log_likelihood_{self.fidelities[0]}", log_lik)
            
            # Add fidelities recursively
            for f in range(1,len(self.fidelities)):
                y_prev_pred_full = self._add_fidelity(model, f, y_prev_pred_full)
            self.model = model
            
            #pm.model_to_graphviz(model)
            self.param_names = [rv.name for rv in self.model.free_RVs]

            self.model = model
    
    def run_inference(self, method="nuts", n_samples=200, n_steps=1000, tune=100, chains=4, target_accept = 0.95,cores=1):
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
                approx = pm.fit(n=n_steps, method="advi", progressbar=True)
                #approx = pm.fit(n=n_steps, method=pm.MeanField(), progressbar=True)
                #approx = pm.fit(n=n_steps, method=pm.MeanField(), obj_optimizer=pm.adam(learning_rate=1e-2), progressbar=True)
                #approx = pm.fit(n=n_steps, method="advi", progressbar=True, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-2)])
                self.trace = approx.sample(draws=n_samples)
                self.elbo = approx.hist
                #plt.plot(approx.hist)
            elif method == "nuts":
                 # HMC Sampling
                self.step = pm.NUTS(target_accept=target_accept)  # Explicitly creating the sampler
                self.trace = pm.sample(n_samples, tune=tune, chains=chains,init="adapt_diag", step=self.step, cores=cores, progressbar=True)
                
            else:
                raise ValueError(f"Unknown inference method: {method}")
            var_names = []
            for f in self.fidelities:
                var_names.append(f"y_likeli_{f}")
            with self.model:
                self.posterior_predictive = pm.sample_posterior_predictive(self.trace, var_names=var_names, return_inferencedata=True)


        return self.trace
    
    def summarize_sampler(self):
        print("Sampler Configuration:")
        #if self.trace.posterior.sizes["chain"] > 1:
        #    print(f"  Step method  : {self.step.__class__.__name__}")
        #    print(f"  Step size    : {self.step.step_size}")
        #    print(f"  Target accept: {self.step.target_accept}")
        #    print()
        print("Posterior Summary:")
        summary=az.summary(self.trace, round_to=3)
        print(summary.to_string())

        n_chains = self.trace.posterior.sizes["chain"]
        n_samples_total = n_chains * self.trace.posterior.sizes["draw"]
        self.table_flat_dict["n_samples"] = n_samples_total

        r_hat_values = summary["r_hat"]
        print("number of r_hat values > 1.05: ", len(r_hat_values[r_hat_values > 1.05]) )
        ess_bulk = summary["ess_bulk"]
        print("number of ess_bulk values < 100: ", len(ess_bulk[ess_bulk < 100]))
        self.table_flat_dict["ess_bulk < 100 [%]"] = f"{len(ess_bulk[ess_bulk < 100])/len(ess_bulk)*100.:.1f}"
        self.table_flat_dict["r_hat > 1.05 [%]"] = f"{len(r_hat_values[r_hat_values > 1.05])/len(r_hat_values)*100.:.1f}"
    
    def plot_trace(self, output_file="trace.png"):
        az.plot_trace(self.trace)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    def plot_diagnostics(self, output_file="diagnostics.png"):
        if self.elbo is None:
            az.plot_energy(self.trace)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.table_flat_dict["BMFI"]=f"".join({f"/{b:.2f}" for b in az.bfmi(self.trace)})[1:]
            return az
        else:
            fig = plt.figure()
            plt.plot(self.elbo)
            plt.xlabel("Iteration")
            plt.ylabel("ELBO")
            plt.title("ADVI Convergence (ELBO)")
            plt.show()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            return fig
    
    def plot_pair(self, output_file="pair_correlations.png"):
        for f in reversed(self.fidelities):
            vars = [v for v in self.param_names if f in v]
            az.rcParams["plot.max_subplots"] = 50
            coffs_var = [v for v in vars if f"coeffs" in v]
            n = self.trace.posterior[coffs_var[0]].shape[-1]+1
            total_plots = n * (n + 1) // 2
            if total_plots <= 50:
                ax = az.plot_pair(self.trace, var_names=vars, kind='kde', divergences=True, marginals=True)
            else:
                ax = az.plot_posterior(self.trace, var_names=["coeffs_lf", "sigma_lf"])
                plt.tight_layout()  
            plt.savefig(f"{output_file[:-4]}_{f}.png", dpi=300, bbox_inches='tight')


    def plot_forrest(self, output_file="forest.png"):
        az.plot_forest(self.trace, var_names=self.param_names)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        return az
    
    def save_trace(self, output_file="trace.nc"):
        az.to_netcdf(self.trace, output_file)
        #az.to_netcdf(self.posterior_predictive,f"{output}/pce_{version}_ppc_nf{len(self.fidelities)}_deg{self.degree}.nc")

    def sanity_check_of_basis(self):
        for f in self.fidelities:
            print("X range:", np.min(self.trainings_data[f][0], axis=0), np.max(self.trainings_data[f][0], axis=0))
            print("y std:", np.std(self.trainings_data[f][1]))
            #print("Basis stds:", np.std(self.basis_matrices[f], axis=0))
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

    def get_coverage(self,trace):
        for f in self.fidelities:
            
            # Stack samples properly: result shape (n_samples, n_observations)
            y_pred_samples = self.posterior_predictive.posterior_predictive[f"y_likeli_{f}"].stack(sample=("chain", "draw")).values  # shape: (n_samples, n_obs)

            # Compute percentiles for each observation
            lower_1sigma = np.percentile(y_pred_samples, 16, axis=1)  # shape (n_obs,)
            upper_1sigma = np.percentile(y_pred_samples, 84, axis=1)  # shape (n_obs,)
            lower_2sigma = np.percentile(y_pred_samples, 2.5, axis=1)  # shape (n_obs,)
            upper_2sigma = np.percentile(y_pred_samples, 97.5, axis=1)  # shape (n_obs,)
            lower_3sigma = np.percentile(y_pred_samples, 0.5, axis=1)  # shape (n_obs,)
            upper_3sigma = np.percentile(y_pred_samples, 99.5, axis=1)  # shape (n_obs,)

            # Your observed y-values
            y_true = self.trainings_data[f][1]  # shape: (n_obs,)

            covered = (y_true >= lower_1sigma) & (y_true <= upper_1sigma)
            coverage_1sigma = np.mean(covered)
            covered = (y_true >= lower_2sigma) & (y_true <= upper_2sigma)
            coverage_2sigma = np.mean(covered)
            covered = (y_true >= lower_3sigma) & (y_true <= upper_3sigma)
            coverage_3sigma = np.mean(covered)

            print(f"Empirical 1σ/2σ/3σ coverage for {f}: {coverage_1sigma*100.:.1f}%/{coverage_2sigma*100.:.1f}%/{coverage_3sigma*100.:.1f}%")

    def find_optimal_order_lf(self, max_order=10, k_folds=5):
        """
        Find the optimal polynomial order for the low-fidelity model using K-fold cross-validation.

        Parameters:
        - max_order (int): Max polynomial order to test
        - k_folds (int): Number of folds in cross-validation

        Returns:
        - optimal_order (int): Best order based on validation error
        """
        lf_key = self.fidelities[0]
        X_lf, y_lf = self.trainings_data[lf_key]
        n_samples = X_lf.shape[0]

        if n_samples < k_folds:
            print(f"Not enough LF samples ({n_samples}) for {k_folds}-fold CV. Using LOOCV instead.")
            kf = KFold(n_splits=n_samples)
        else:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        errors = []

        for order in range(1, max_order + 1):
            fold_errors = []
            for train_idx, test_idx in kf.split(X_lf):
                X_train, X_test = X_lf[train_idx], X_lf[test_idx]
                y_train, y_test = y_lf[train_idx], y_lf[test_idx]

                # Create Legendre basis
                Phi_train = self.multivariate_legendre_with_interactions(order, X_train)
                Phi_test = self.multivariate_legendre_with_interactions(order, X_test)

                # Fit coefficients
                coeffs = np.linalg.lstsq(Phi_train, y_train, rcond=None)[0]

                # Predict
                y_pred = Phi_test @ coeffs
                fold_errors.append(mean_squared_error(y_test, y_pred))

            avg_error = np.mean(fold_errors)
            errors.append(avg_error)
            print(f"Order {order} → CV MSE: {avg_error:.4f}")

        optimal_order = np.argmin(errors) + 1
        print(f"Optimal LF order: {optimal_order}")
        return optimal_order

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
        n_splits,_= self.trainings_data[self.fidelities[0]][0].shape
        max_order = min(5, n_splits)

        errors = []
        for order in range(1, max_order + 1):
            # Generate basis for all fidelities
            basis_with_interactions = {}
            c = {}
            for fidelity in self.fidelities:
                basis_with_interactions[fidelity] = self.multivariate_legendre_with_interactions(
                    order, self.trainings_data[fidelity][0]
                )
                c[fidelity] = np.linalg.lstsq(
                    basis_with_interactions[fidelity],
                    self.trainings_data[fidelity][1],
                    rcond=None
                )[0]

            y_pred = {}
            for i, fidelity in enumerate(self.fidelities[1:]):
                n_samples = self.trainings_data[fidelity][0].shape[0]
                if n_samples <= 10:
                    print(f"Not enough high-fidelity samples ({n_samples}) for cross-validation with {fidelity}.")
                    continue
                f_prev = self.fidelities[i]  # lower fidelity
                f_curr = fidelity            # current higher fidelity

                # Predict LF contribution on HF inputs
                y_lf_on_hf = basis_with_interactions[f_curr] @ c[f_prev]

                # Fit optimal rho_hat: f_HF ≈ rho * f_LF
                rho_hat = np.linalg.lstsq(
                    y_lf_on_hf.reshape(-1, 1),
                    self.trainings_data[f_curr][1],
                    rcond=None
                )[0][0]  # take scalar value

                # Compute discrepancy
                delta = self.trainings_data[f_curr][1] - rho_hat * y_lf_on_hf

                mse_fold = []
                x_train = {}
                y_train = {}
                x_test = {}
                y_test = {}

                kf = KFold(n_splits=n_samples)
                for train_idx, test_idx in kf.split(self.trainings_data[f_curr][0]):
                    x_train[f_curr], x_test[f_curr] = (
                        self.trainings_data[f_curr][0][train_idx],
                        self.trainings_data[f_curr][0][test_idx]
                    )
                    y_train[f_curr], y_test[f_curr] = (
                        self.trainings_data[f_curr][1][train_idx],
                        self.trainings_data[f_curr][1][test_idx]
                    )

                    # Generate basis for train and test
                    phi_train = self.multivariate_legendre_with_interactions(order, x_train[f_curr])
                    phi_test = self.multivariate_legendre_with_interactions(order, x_test[f_curr])

                    # Also get LF prediction on test input
                    basis_lf_test = self.multivariate_legendre_with_interactions(order, x_test[f_curr])
                    y_lf_test = basis_lf_test @ c[f_prev]
                    y_lf_test_scaled = rho_hat * y_lf_test

                    # Train discrepancy model on training data
                    y_lf_train = basis_with_interactions[f_curr][train_idx] @ c[f_prev]
                    delta_train = y_train[f_curr] - rho_hat * y_lf_train
                    c_hf = np.linalg.lstsq(phi_train, delta_train, rcond=None)[0]

                    # Predict delta on test
                    delta_pred_test = phi_test @ c_hf
                    y_pred_fold = y_lf_test_scaled + delta_pred_test

                    mse_fold.append(mean_squared_error(y_test[f_curr], y_pred_fold))

                # Store average error for this order
                errors.append(np.mean(mse_fold))
        if errors ==[]:
            return self.find_optimal_order_lf()
        # Return the optimal order (1-based indexing for order)
        optimal_order = np.argmin(errors) + 1
        print(f"The optimal order is {optimal_order}")
        return optimal_order

    def add_log_likelihood_manually(self):
        if "log_likelihood" not in self.trace.groups():
            ll = self.trace.posterior["log_likelihood_lf"]
        ll = self.trace.posterior["log_likelihood_lf"]

    def pareto_k(self, output_file="pareto_k_diagnostics.png"):
        if "log_likelihood" not in self.trace.groups():
            self.trace.add_groups({"log_likelihood": {"log_likelihood_lf": self.trace.posterior["log_likelihood_lf"]}})
            ll = self.trace.posterior["log_likelihood_lf"]

        # Compute loo
        loo_result = az.loo(self.trace, pointwise=True)

        # Access p_loo
        p_loo = loo_result.p_loo
        print("Effective number of parameters (p_loo):", p_loo)
        self.table_flat_dict["p_loo"] = f"{p_loo:.1f}"

        # Similarly for WAIC
        waic_result = az.waic(self.trace, pointwise=True)
        p_waic = waic_result.p_waic
        print("Effective number of parameters (p_waic):", p_waic)
        self.table_flat_dict["p_waic"] = f"{p_waic:.1f}"

        loo_result = az.loo(self.trace, pointwise=True)
        # Pareto k values are in loo_result.pareto_k
        pareto_k = loo_result.pareto_k
        n_bad = (pareto_k > 0.5).sum().item()
        pct_bad = 100 * n_bad / pareto_k.size
        print(f"{n_bad} out of {pareto_k.size} Pareto k values are > 0.5 ({pct_bad:.1f}%)")
        self.table_flat_dict["pareto_k > 0.5 [%]"] = f"{pct_bad:.1f}"

        fig = plt.figure(figsize=(10,6))
        plt.plot(pareto_k, marker='o', linestyle='none')
        plt.axhline(0.7, color='red', linestyle='--', label='Warning threshold (0.7)')
        plt.axhline(0.5, color='orange', linestyle='--', label='Caution threshold (0.5)')
        plt.xlabel('Data point index')
        plt.ylabel('Pareto k')
        plt.title('Pareto k diagnostics for LOO')
        plt.legend()
        plt.show()
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        return fig

    def readin_trace(self, input_file="trace.nc"):
        self.trace = az.from_netcdf(input_file)

    def set_table(self):
        # Flatten the dictionary

        self.table_flat_dict = {
            f"{fidelity}_{key}": value
            for fidelity, subdict in self.priors.items()
            for key, value in subdict.items()
        }
        self.table_flat_dict["polynomial order"] = "/".join(f"{v}" for v in self.degree)
        
    def print_table(self, output_file="table.csv"):
        # Create one-row DataFrame
        df = pd.DataFrame([self.table_flat_dict])
        display(df)
        df.to_csv(output_file, index=False)
        # Set a fixed column width for alignment (adjust as needed)
        #col_width = 26

        # Format and print aligned table
        #header = " | ".join(f"{col:<{col_width}}" for col in df.columns)
        #separator = "-+-".join("-" * col_width for _ in df.columns)
        #row = " | ".join(f"{str(val):<{col_width}}" for val in df.iloc[0])

        #print(header)
        #print(separator)
        #print(row)

        