import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import GPy
from emukit.multi_fidelity import kernels
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_x_list_to_array,
    convert_xy_lists_to_arrays
)
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.acquisition import Acquisition
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.core.loop.candidate_point_calculators import SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
import copy

# Ensure reproducibility
np.random.seed(123)


class MFGPModel():
    def __init__(self, trainings_data, noise, inequality_constraints=None):
        self.trainings_data = copy.deepcopy(trainings_data)
        self.fidelities = list(self.trainings_data.keys())
        self.nfidelities = len(self.fidelities)
        self.noise = noise
        self.model = None
        if inequality_constraints==None:
            self.inequality_constraints=MFGPInequalityConstraints()
        else:
            self.inequality_constraints=inequality_constraints

    def set_traings_data(self, trainings_data):
        self.trainings_data = copy.deepcopy(trainings_data)

    def build_model(self,n_restarts=10, custom_lengthscale=None):
        """
        Constructs and trains a linear multi-fidelity model using Gaussian processes.
        """
        x_train = []
        y_train = []
        for fidelity in self.fidelities:
            x_tmp=np.atleast_2d(self.trainings_data[fidelity][0])
            y_tmp=np.atleast_2d(self.trainings_data[fidelity][1]).T
            x_train.append(x_tmp)
            y_train.append(y_tmp)
        
        X_train, Y_train = convert_xy_lists_to_arrays(x_train, y_train)
        
        # Add diagnostic information
        print(f"Training data shapes:")
        for i, fidelity in enumerate(self.fidelities):
            print(f"  Fidelity {fidelity}: X={x_train[i].shape}, Y={y_train[i].shape}")
        print(f"Combined X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        
        # Define kernels for each fidelity
        kernels_list = []
        
        # FIX: Use correct input dimension (number of features, not number of samples)
        input_dim = X_train.shape[1] - 1  # Subtract 1 for fidelity indicator
        print(f"Input dimension for kernels: {input_dim}")
        
        # Auto-initialize lengthscales if not provided
        if custom_lengthscale is None:
            auto_lengthscales = self.get_auto_lengthscales()
            if auto_lengthscales is not None:
                # Use median of parameter ranges as default lengthscale
                default_lengthscale = float(np.median(auto_lengthscales))
                print(f"Auto-computed lengthscale: {default_lengthscale}")
            else:
                default_lengthscale = 1.0
                print(f"Using default lengthscale: {default_lengthscale}")
        else:
            default_lengthscale = custom_lengthscale
            print(f"Using custom lengthscale: {custom_lengthscale}")

        for f in range(self.nfidelities - 1):
            rbf1 = GPy.kern.RBF(input_dim=input_dim, name=f"RBF_rho_{f}")
            rbf2 = GPy.kern.RBF(input_dim=1, name=f"RBF_delta_{f}")

            # Set lengthscales with bounds to prevent numerical issues
            rbf1.lengthscale = default_lengthscale
            rbf2.lengthscale = 1.0  # For fidelity dimension
            
            # Add bounds to prevent extreme values
            max_lengthscale = default_lengthscale * 10  # Maximum 10x the data range
            min_lengthscale = default_lengthscale * 0.1  # Minimum 0.1x the data range
            
            rbf1.lengthscale.constrain_bounded(min_lengthscale, max_lengthscale)
            rbf2.lengthscale.constrain_bounded(0.1, 10.0)  # Reasonable bounds for fidelity
            
            # Also constrain variances to prevent numerical issues
            rbf1.variance.constrain_bounded(1e-6, 10.0)
            rbf2.variance.constrain_bounded(1e-6, 10.0)
            
            print(f"  Kernel {f}: lengthscale bounds [{min_lengthscale:.2f}, {max_lengthscale:.2f}]")

            kernels_list.append(rbf1)
            kernels_list.append(rbf2)


        lin_mf_kernel = kernels.LinearMultiFidelityKernel(kernels_list)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=len(self.fidelities))

        # Fix noise terms for each fidelity with validation
        print(f"Setting noise levels:")
        for i,fidelity in enumerate(self.fidelities):
            noise_val = self.noise[fidelity]
            print(f"  Fidelity {fidelity}: original noise = {noise_val}")
            
            # Fix zero noise problem - ensure minimum noise level
            min_noise = 1e-6  # Minimum noise to prevent numerical issues
            if noise_val <= min_noise:
                noise_val = max(min_noise, 1e-4)  # Use small but non-zero noise
                print(f"  WARNING: Zero/low noise detected! Setting minimum noise = {noise_val}")
                # Update the noise dictionary for consistency
                self.noise[fidelity] = noise_val
            
            # Validate noise values
            if noise_val > 1.0:
                print(f"WARNING: Very high noise value {noise_val} for fidelity {fidelity}")
            
            print(f"  Fidelity {fidelity}: final noise = {noise_val}")
            
            # Construct the attribute name dynamically
            noise_attr = f"Gaussian_noise" if i == 0 else f"Gaussian_noise_{i}"
            try:
                getattr(gpy_lin_mf_model.mixed_noise, noise_attr).fix(noise_val)
            except AttributeError:
                print(f"Error: Attribute '{noise_attr}' not found in the model.")
                raise


        # Wrap and optimize the model
        print(f"Starting optimization with {n_restarts} restarts...")
        self.model = GPyMultiOutputWrapper(
            gpy_lin_mf_model, len(self.fidelities), n_optimization_restarts=n_restarts, verbose_optimization=True
        )
        
        # Store initial likelihood - fix AttributeError
        try:
            initial_likelihood = self.model.gpy_model.log_likelihood()
            print(f"Initial log likelihood: {initial_likelihood}")
        except:
            print("Could not access initial likelihood - proceeding with optimization")
            initial_likelihood = None
        
        self.model.optimize()
        
        # Store final likelihood
        try:
            final_likelihood = self.model.gpy_model.log_likelihood()
            print(f"Final log likelihood: {final_likelihood}")
            if initial_likelihood is not None:
                print(f"Likelihood improvement: {final_likelihood - initial_likelihood}")
        except:
            print("Could not access final likelihood")
        
        # Print final kernel parameters
        print(f"Optimized model parameters:")
        try:
            print(self.model.gpy_model)
        except:
            print("Could not display model parameters")
        
        return self.model

    def set_data(self,trainings_data_new):
        x_train = []
        y_train = []
        for fidelity in self.fidelities:
            self.trainings_data[fidelity][0].extend(trainings_data_new[fidelity][0])
            self.trainings_data[fidelity][1].extend(trainings_data_new[fidelity][1])
            x_tmp=np.atleast_2d(self.trainings_data[fidelity][0])
            y_tmp=np.atleast_2d(self.trainings_data[fidelity][1]).T
            x_train.append(x_tmp)
            y_train.append(y_tmp)
        
        X_train, Y_train = convert_xy_lists_to_arrays(x_train, y_train)
        self.model.set_data(X_train, Y_train)

    def max_acquisition_integrated_variance_reduction(self, parameters):
        ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)

        optimizer = GradientAcquisitionOptimizer(parameter_space)
        multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(optimizer, parameter_space)
        #acquisition = ModelVariance(mf_model) * inequality_constraints
        acquisition = IntegratedVarianceReduction(self.model, parameter_space, num_monte_carlo_points=2000) * self.inequality_constraints

        # Create batch candidate point calculator
        sequential_point_calculator = SequentialPointCalculator(acquisition, multi_source_acquisition_optimizer)
        loop_state = create_loop_state(self.model.X, self.model.Y)
        x_next = sequential_point_calculator.compute_next_points(loop_state)

        return x_next, acquisition
    
    def max_acquisition_multisource(self, parameters):
        ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)

        optimizer = GradientAcquisitionOptimizer(parameter_space)
        us_acquisition = MultiInformationSourceEntropySearch(self.model, parameter_space) * self.inequality_constraints
        x_new, _ = optimizer.optimize(us_acquisition)
        return x_new, us_acquisition
    
    def max_acquisition_model_variance(self, parameters):
        ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)


        optimizer = GradientAcquisitionOptimizer(parameter_space)
        multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(optimizer, parameter_space)
        acquisition = ModelVariance(self.model) * self.inequality_constraints

        # Create batch candidate point calculator
        sequential_point_calculator = SequentialPointCalculator(acquisition, multi_source_acquisition_optimizer)
        loop_state = create_loop_state(self.model.X, self.model.Y)
        x_next = sequential_point_calculator.compute_next_points(loop_state)
        
        return x_next, acquisition

    def evaluate_model(self, x, fidelity=2):
        x_eval=np.array([x])
        SPLIT = 1
        X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
        return self.model.predict(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])[0][0][0]

    def evaluate_model_gradient(self, x, fidelity=2):
        x_eval=np.array([x])
        SPLIT = 1
        X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
        return self.model.get_prediction_gradients(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])[0][0]

    def evaluate_model_uncertainty(self, x, fidelity=2):
        x_eval=np.array([x])
        SPLIT = 1
        X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
        _, var = self.model.predict(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])
        var=var[0][0]
        var=np.sqrt(var)
        return var

    def get_min(self, parameters, x0=None, fidelity=2):

        def f(x):
            self.evaluate_model(x, fidelity)

        bnds=[]
        for i in parameters:
            bnds.append((parameters[i][0],parameters[i][1]))
            if x0==None:
                x0.append((parameters[i][1]-parameters[i][0])/2.)
        x0=np.array(x0)
        
        res = minimize(f, x0,bounds=bnds)
        return res.x, res.fun
    
    def get_min_constrained(self, parameters, fidelity=2):
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)

        model = MFGPAuxilaryModel(self, fidelity)

        optimizer = GradientAcquisitionOptimizer(parameter_space)
        acquisition = model
        x_min, _ = optimizer.optimize(acquisition)
        x_min=[x for x in x_min[0]]
        return x_min, self.evaluate_model(x_min, fidelity)
    
    def analyze_training_data(self):
        """
        Analyze training data to diagnose potential issues with uncertainty bands
        """
        print("\n=== TRAINING DATA ANALYSIS ===")
        
        for fidelity in self.fidelities:
            X = np.array(self.trainings_data[fidelity][0])
            Y = np.array(self.trainings_data[fidelity][1])
            
            print(f"\nFidelity {fidelity}:")
            print(f"  Number of samples: {len(X)}")
            print(f"  Input dimension: {X.shape[1] if X.ndim > 1 else 1}")
            
            # Analyze input space coverage
            if X.ndim > 1:
                for i in range(X.shape[1]):
                    col = X[:, i]
                    print(f"    Feature {i}: min={col.min():.4f}, max={col.max():.4f}, std={col.std():.4f}")
                    
                    # Check for poor coverage
                    unique_vals = len(np.unique(col))
                    if unique_vals < len(col) * 0.1:  # Less than 10% unique values
                        print(f"    WARNING: Feature {i} has poor diversity ({unique_vals} unique values)")
            
            # Analyze output space
            print(f"  Output: min={Y.min():.4f}, max={Y.max():.4f}, std={Y.std():.4f}")
            
            # Check for data scaling issues
            if Y.std() > 10 or Y.std() < 0.01:
                print(f"  WARNING: Output has unusual scale (std={Y.std():.6f})")
                
        # Check fidelity correlations
        if len(self.fidelities) > 1:
            print(f"\n=== FIDELITY CORRELATION ANALYSIS ===")
            fidelity_names = list(self.fidelities)
            
            for i in range(len(fidelity_names)):
                for j in range(i+1, len(fidelity_names)):
                    fid1, fid2 = fidelity_names[i], fidelity_names[j]
                    Y1 = np.array(self.trainings_data[fid1][1])
                    Y2 = np.array(self.trainings_data[fid2][1])
                    
                    # Simple correlation if same number of points
                    if len(Y1) == len(Y2):
                        corr = np.corrcoef(Y1.flatten(), Y2.flatten())[0,1]
                        print(f"  Correlation {fid1}-{fid2}: {corr:.3f}")
                        if abs(corr) < 0.3:
                            print(f"    WARNING: Low correlation between fidelities!")
    
    def get_auto_lengthscales(self):
        """
        Automatically compute reasonable lengthscales based on data spread
        """
        all_X = []
        for fidelity in self.fidelities:
            X = np.array(self.trainings_data[fidelity][0])
            all_X.append(X)
        
        if all_X:
            X_combined = np.vstack(all_X)
            if X_combined.ndim > 1:
                # Compute range for each dimension
                param_ranges = X_combined.max(axis=0) - X_combined.min(axis=0)
                # Use fraction of range as lengthscale (common heuristic)
                auto_lengthscales = param_ranges * 0.3  # 30% of range
                return auto_lengthscales
        
        return None

    def build_model_alternative(self, n_restarts=10, use_independent_kernels=False):
        """
        Alternative model building approach for cases where linear multi-fidelity fails
        """
        x_train = []
        y_train = []
        for fidelity in self.fidelities:
            x_tmp=np.atleast_2d(self.trainings_data[fidelity][0])
            y_tmp=np.atleast_2d(self.trainings_data[fidelity][1]).T
            x_train.append(x_tmp)
            y_train.append(y_tmp)
        
        X_train, Y_train = convert_xy_lists_to_arrays(x_train, y_train)
        
        print(f"Using alternative model building approach...")
        print(f"Combined X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        
        if use_independent_kernels:
            # Use independent RBF kernels for each fidelity (no correlation assumption)
            print("Building independent kernels per fidelity...")
            
            input_dim = X_train.shape[1] - 1
            kernels_list = []
            
            # Get auto lengthscale
            auto_lengthscales = self.get_auto_lengthscales()
            default_lengthscale = float(np.median(auto_lengthscales)) if auto_lengthscales is not None else 1.0
            
            for f in range(self.nfidelities):
                rbf = GPy.kern.RBF(input_dim=input_dim, name=f"RBF_fidelity_{f}")
                rbf.lengthscale = default_lengthscale
                rbf.lengthscale.constrain_bounded(default_lengthscale * 0.1, default_lengthscale * 10)
                rbf.variance.constrain_bounded(1e-6, 10.0)
                kernels_list.append(rbf)
            
            # Create multi-output kernel
            from GPy.kern import Matern52
            kernel = GPy.util.multioutput.ICM(input_dim=input_dim, num_outputs=self.nfidelities, 
                                           kernel=Matern52(input_dim))
            
        else:
            # Use simpler linear multi-fidelity with better initialization
            print("Building simplified linear multi-fidelity...")
            kernels_list = []
            input_dim = X_train.shape[1] - 1
            
            # Get auto lengthscale
            auto_lengthscales = self.get_auto_lengthscales()
            default_lengthscale = float(np.median(auto_lengthscales)) if auto_lengthscales is not None else 1.0
            
            # Use Matern kernels instead of RBF for better behavior
            for f in range(self.nfidelities - 1):
                kern1 = GPy.kern.Matern52(input_dim=input_dim, name=f"Matern_rho_{f}")
                kern2 = GPy.kern.Matern52(input_dim=1, name=f"Matern_delta_{f}")
                
                kern1.lengthscale = default_lengthscale
                kern2.lengthscale = 1.0
                
                # Conservative bounds
                kern1.lengthscale.constrain_bounded(default_lengthscale * 0.5, default_lengthscale * 2)
                kern2.lengthscale.constrain_bounded(0.5, 2.0)
                kern1.variance.constrain_bounded(0.01, 1.0)
                kern2.variance.constrain_bounded(0.01, 1.0)
                
                kernels_list.append(kern1)
                kernels_list.append(kern2)
            
            kernel = kernels.LinearMultiFidelityKernel(kernels_list)
        
        # Build model
        if use_independent_kernels:
            # Independent GP for each fidelity
            models = []
            for i, fidelity in enumerate(self.fidelities):
                X_fid = x_train[i]
                Y_fid = y_train[i]
                
                noise_val = max(self.noise[fidelity], 1e-4)
                
                model_fid = GPy.models.GPRegression(X_fid, Y_fid, kernels_list[i])
                model_fid.Gaussian_noise.variance.fix(noise_val)
                model_fid.optimize_restarts(n_restarts//2)
                models.append(model_fid)
            
            # Store models for prediction
            self.independent_models = models
            self.model = None  # Signal we're using independent models
            print("Built independent models for each fidelity")
            
        else:
            # Standard multi-fidelity approach
            gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, kernel, n_fidelities=len(self.fidelities))
            
            # Set noise
            for i,fidelity in enumerate(self.fidelities):
                noise_val = max(self.noise[fidelity], 1e-4)
                noise_attr = f"Gaussian_noise" if i == 0 else f"Gaussian_noise_{i}"
                getattr(gpy_lin_mf_model.mixed_noise, noise_attr).fix(noise_val)
            
            self.model = GPyMultiOutputWrapper(gpy_lin_mf_model, len(self.fidelities), 
                                            n_optimization_restarts=n_restarts, verbose_optimization=True)
            self.model.optimize()
            
        return self.model if hasattr(self, 'model') and self.model else self.independent_models

class MFGPAuxilaryModel(Acquisition):
    def __init__(self, mf_model, fidelity):
        self.mf_model = mf_model
        self.fidelity = fidelity
        self.inequality=self.mf_model.inequality_constraints

    def evaluate(self, x):
        delta_inequ=self.inequality.evaluate(x)
        delta_inequ[delta_inequ == 0] = np.inf
        delta_x = np.ones(len(x))
        for i,xi in enumerate(x[:,:]):
            delta_x[i] = -1.*self.mf_model.evaluate_model(xi, self.fidelity)
            if self.mf_model.evaluate_model(xi, self.fidelity) <= 0.:
                delta_x[i] = -0.00001
        return delta_x[:, None]*delta_inequ[:,None]
    
    @property
    def has_gradients(self):
        return True
    
    def get_gradients(self,x):
        delta_x = np.ones(len(x))
        for i,xi in enumerate(x[:,:]):
            delta_x[i] = self.mf_model.evaluate_model_gradient(xi,self.fidelity)[0][0]
        return delta_x[:, None]

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x[:,:].shape)

# --- Custom Acquisitions ---
class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        return np.array([self.costs[i] for i in fidelity_index])[:, None]

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)


class MFGPInequalityConstraints(Acquisition):
    def __init__(self):
        pass

    def evaluate(self, x):
        delta_x = np.ones(len(x))

        return delta_x[:, None]


    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x[:, :-1].shape)
