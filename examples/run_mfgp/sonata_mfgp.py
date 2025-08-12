# %% [markdown]
# ## Followed this notebook: https://github.com/EmuKit/emukit/blob/main/notebooks/Emukit-tutorial-multi-fidelity.ipynb

# %%
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import importlib.util
import random
random.seed(42)

import GPy
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

# %%
# Configuration
FIGURE_SUFFIX = "_150bsp"  # Suffix for all saved figures and files

with open("../coherent/settings.yaml", "r") as f:
    config_file = yaml.safe_load(f)

version       = config_file["path_settings"]["version"]
path_out_cnp  = config_file["path_settings"]["path_out_cnp"]
path_out_mfgp = config_file["path_settings"]["path_out_mfgp"]
file_in=f'{path_out_cnp}/cnp_{version}_output_20epochs{FIGURE_SUFFIX}.csv'

# %%
# data processing/setup
np.random.seed(42)

if not os.path.exists(path_out_mfgp):
   os.makedirs(path_out_mfgp)

# Set parameter name/x_labels -> needs to be consistent with data input file
x_labels        = config_file["simulation_settings"]["theta_headers"]
y_label_cnp     = 'y_cnp'
y_err_label_cnp = 'y_cnp_err'
y_label_sim     = 'y_raw'

# %%
data=pd.read_csv(file_in)

LF_cnp_noise=np.mean(data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_err_label_cnp].to_numpy())
HF_cnp_noise=np.mean(data.loc[(data['fidelity']==1.) & (data['iteration']==0)][y_err_label_cnp].to_numpy())
LF_sim_noise=np.std(data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_label_sim].to_numpy())
HF_sim_noise=np.std(data.loc[(data['fidelity']==1.) & (data['iteration']==0)][y_label_sim].to_numpy())

#x_train_l, x_train_h, y_train_l, y_train_h = ([],[],[],[])
#row_h=data.index[data['fidelity'] == 1].tolist()
#row_l=data.index[data['fidelity'] == 0].tolist()

#x_train_hf_sim = data.loc[(data['fidelity']==1.) & (data['iteration']==0)][x_labels].to_numpy().tolist()
#y_train_hf_sim = data.loc[(data['fidelity']==1.) & (data['iteration']==0)][y_label_sim].to_numpy().tolist()

#x_train_hf_cnp = data.loc[(data['fidelity']==1.) & (data['iteration']==0)][x_labels].to_numpy().tolist()
#y_train_hf_cnp = data.loc[(data['fidelity']==1.) & (data['iteration']==0)][y_label_cnp].to_numpy().tolist()

#x_train_lf_sim = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][x_labels].to_numpy().tolist()
#y_train_lf_sim = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_label_sim].to_numpy().tolist()


# Get the filtered dataframe first
filtered_data = data.loc[(data['fidelity']==1.) & (data['iteration']==0)]

# Get unique combinations of x_label values to select diverse training points
unique_x_combinations = filtered_data[x_labels].drop_duplicates()

# Select up to 3 diverse training points based on different x_label combinations
if len(unique_x_combinations) >= 3:
    # Select 3 points with most diverse x values
    # Get min, max, and a middle point for each x dimension
    x_data = unique_x_combinations.values
    
    # Find points with min/max values for first dimension (water_shielding_mm)
    min_x1_idx = np.argmin(x_data[:, 0])
    max_x1_idx = np.argmax(x_data[:, 0])
    
    # Find a point with different x2 value (veto_thickness_mm) that's not min/max x1
    remaining_indices = [i for i in range(len(x_data)) if i not in [min_x1_idx, max_x1_idx]]
    if remaining_indices:
        # Select point with most different x2 value from the min/max x1 points
        x1_values = x_data[[min_x1_idx, max_x1_idx], 1]
        mid_x2_idx = remaining_indices[np.argmax([abs(x_data[i, 1] - np.mean(x1_values)) for i in remaining_indices])]
    else:
        mid_x2_idx = min_x1_idx if min_x1_idx != max_x1_idx else 0
    
    selected_combinations = unique_x_combinations.iloc[[min_x1_idx, max_x1_idx, mid_x2_idx]]
else:
    # If fewer than 3 unique combinations, use all available
    selected_combinations = unique_x_combinations

# Find the indices in filtered_data that match these selected combinations
train_indices = []
for _, combo in selected_combinations.iterrows():
    # Find first occurrence of this combination in filtered_data
    mask = (filtered_data[x_labels[0]] == combo[x_labels[0]]) & (filtered_data[x_labels[1]] == combo[x_labels[1]])
    matching_indices = filtered_data[mask].index.tolist()
    if matching_indices:
        train_indices.append(matching_indices[0])  # Take first occurrence

test_indices = filtered_data.index.difference(train_indices)



filtered_data = data.loc[(data['fidelity']==1.) & (data['iteration']==0)]

unique_x_combinations = filtered_data[x_labels].drop_duplicates().values

combination_1 = []
combination_2 = []
combination_3 = []

samples_with_combination_1 = filtered_data.loc[filtered_data[x_labels].values==unique_x_combinations[0]]
combination_1.extend(list(set(samples_with_combination_1.index.to_list())))
samples_with_combination_2 = filtered_data.loc[filtered_data[x_labels].values==unique_x_combinations[1]]
combination_2.extend(list(set(samples_with_combination_2.index.to_list())))
samples_with_combination_3 = filtered_data.loc[filtered_data[x_labels].values==unique_x_combinations[2]]
combination_3.extend(list(set(samples_with_combination_3.index.to_list())))

random.shuffle(combination_1)
random.shuffle(combination_2)
random.shuffle(combination_3)

combination_1_70 = combination_1[:int(len(combination_1) // (10/9))]
combination_1_30 = combination_1[int(len(combination_1) // (10/9)):]
combination_2_70 = combination_2[:int(len(combination_2) // (10/9))]
combination_2_30 = combination_2[int(len(combination_2) // (10/9)):]
combination_3_70 = combination_3[:int(len(combination_3) // (10/9))]
combination_3_30 = combination_3[int(len(combination_3) // (10/9)):]


# # Extract training data
# x_train_hf_sim = filtered_data.loc[train_indices][x_labels].to_numpy().tolist()
# y_train_hf_sim = filtered_data.loc[train_indices][y_label_sim].to_numpy().tolist()

# # Extract testing data
# x_test_hf_sim = filtered_data.loc[test_indices][x_labels].to_numpy().tolist()
# y_test_hf_sim = filtered_data.loc[test_indices][y_label_sim].to_numpy().tolist()

# Approach 2
# # Extract training data
# x_train_hf_sim = filtered_data[x_labels].to_numpy().tolist()
# y_train_hf_sim = filtered_data[y_label_sim].to_numpy().tolist()

# # Extract testing data
# x_test_hf_sim = filtered_data[x_labels].to_numpy().tolist()
# y_test_hf_sim = filtered_data[y_label_sim].to_numpy().tolist()

## Approach 3
# Extract training data
x_train_hf_sim = filtered_data.loc[combination_1_70][x_labels].to_numpy().tolist()
x_train_hf_sim.extend(filtered_data.loc[combination_2_70][x_labels].to_numpy().tolist())
x_train_hf_sim.extend(filtered_data.loc[combination_3_70][x_labels].to_numpy().tolist())
y_train_hf_sim = filtered_data.loc[combination_1_70][y_label_sim].to_numpy().tolist()
y_train_hf_sim.extend(filtered_data.loc[combination_2_70][y_label_sim].to_numpy().tolist())
y_train_hf_sim.extend(filtered_data.loc[combination_3_70][y_label_sim].to_numpy().tolist())
combined_train_hf_sim = list(zip(x_train_hf_sim, y_train_hf_sim))
random.shuffle(combined_train_hf_sim)
x_train_hf_sim, y_train_hf_sim = zip(*combined_train_hf_sim)
x_train_hf_sim = list(x_train_hf_sim)
y_train_hf_sim = list(y_train_hf_sim)

# Extract testing data
x_test_hf_sim = filtered_data.loc[combination_1_30][x_labels].to_numpy().tolist()
x_test_hf_sim.extend(filtered_data.loc[combination_2_30][x_labels].to_numpy().tolist())
x_test_hf_sim.extend(filtered_data.loc[combination_3_30][x_labels].to_numpy().tolist())
y_test_hf_sim = filtered_data.loc[combination_1_30][y_label_sim].to_numpy().tolist()
y_test_hf_sim.extend(filtered_data.loc[combination_2_30][y_label_sim].to_numpy().tolist())
y_test_hf_sim.extend(filtered_data.loc[combination_3_30][y_label_sim].to_numpy().tolist())
combined_test_hf_sim = list(zip(x_test_hf_sim, y_test_hf_sim))
random.shuffle(combined_test_hf_sim)
x_test_hf_sim, y_test_hf_sim = zip(*combined_test_hf_sim)
x_test_hf_sim = list(x_test_hf_sim)
y_test_hf_sim = list(y_test_hf_sim)

x_train_lf_cnp = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][x_labels].to_numpy().tolist()
y_train_lf_cnp = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_label_cnp].to_numpy().tolist()


trainings_data = {"lf": [x_train_lf_cnp,y_train_lf_cnp], "hf": [x_train_hf_sim,y_train_hf_sim]}#, } "mf": [x_train_hf_cnp,y_train_hf_cnp]
noise = {"lf": LF_cnp_noise, "hf": HF_sim_noise*0.001}#, "hf": 0.0}  # why were mf and hf noise originally set to 0?
# noise = {"lf": 1.7e-6, "hf": 1.7e-6}

# %%
fidelities = list(trainings_data.keys())
nfidelities = len(fidelities)

# %%
x_train = []
y_train = []
for fidelity in fidelities:
    x_tmp=np.atleast_2d(trainings_data[fidelity][0])
    y_tmp=np.atleast_2d(trainings_data[fidelity][1]).T
    x_train.append(x_tmp)
    y_train.append(y_tmp)

X_train, Y_train = convert_xy_lists_to_arrays(x_train, y_train)

# %%
num_fidelities = 2  # just lf and hf for now
kernels = [GPy.kern.Matern32(input_dim=X_train[0].shape[0] - 1), GPy.kern.Matern32(input_dim=X_train[0].shape[0] - 1)]  # since there are two theta parameters, input_dim is 2

linear_mf_kernel = LinearMultiFidelityKernel(kernels)
gpy_linear_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, linear_mf_kernel, n_fidelities = num_fidelities)

# set noise
gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(noise['lf'])  # lf noise
gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(noise['hf'])  # mf/hf noise

# %%
# SET KERNEL HYPERPARAMETER BOUNDS
# Low-fidelity kernel (Mat32)
gpy_linear_mf_model['multifidelity.Mat32.variance'].constrain_bounded(1e-6, 1e2)
gpy_linear_mf_model['multifidelity.Mat32.lengthscale'].constrain_bounded(1e-2, 1e3)

# High-fidelity kernel (Mat32_1) 
gpy_linear_mf_model['multifidelity.Mat32_1.variance'].constrain_bounded(1e-6, 1e2)
gpy_linear_mf_model['multifidelity.Mat32_1.lengthscale'].constrain_bounded(1e-2, 1e3)

# Scale parameter (correlation between fidelities)
gpy_linear_mf_model['multifidelity.scale'].constrain_bounded(1e-3, 1e1)

# If you can unfix noise, increase it slightly
if hasattr(gpy_linear_mf_model.mixed_noise.Gaussian_noise, 'unfix'):
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.unfix()
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.unfix()
    
    # Set to 1.2x your original noise values with some bounds
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.constrain_bounded(
        noise['lf'] * 0.99, noise['lf'] * 1.01)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.constrain_bounded(
        noise['hf'] * 0.99, noise['hf'] * 1.01)

# %%
'''# More aggressive bounds that encourage higher uncertainty
gpy_linear_mf_model['multifidelity.Mat32.variance'].constrain_bounded(1e-5, 1e3)
gpy_linear_mf_model['multifidelity.Mat32_1.variance'].constrain_bounded(1e-5, 1e3)
gpy_linear_mf_model['multifidelity.Mat32.lengthscale'].constrain_bounded(1e-4, 1e2)
gpy_linear_mf_model['multifidelity.Mat32_1.lengthscale'].constrain_bounded(1e-4, 1e2)

# Allow the scale parameter more freedom
gpy_linear_mf_model['multifidelity.scale'].constrain_bounded(1e-2, 1e2)

# If you can unfix noise, increase it slightly
if hasattr(gpy_linear_mf_model.mixed_noise.Gaussian_noise, 'unfix'):
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.unfix()
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.unfix()
    
    # Set to 1.2x your original noise values with some bounds
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.constrain_bounded(
        noise['lf'] * 0.85, noise['lf'] * 1.15)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.constrain_bounded(
        noise['hf'] * 0.85, noise['hf'] * 1.15)'''

# %%
## Wrap the model using the given 'GPyMultiOutputWrapper'
lin_mf_model = GPyMultiOutputWrapper(gpy_linear_mf_model, num_fidelities, n_optimization_restarts=10, verbose_optimization=True)

## Fit the model
lin_mf_model.optimize()

# %%
x_plot = np.linspace(0, 120, 200)[:, None]
x_plot2 = np.hstack((x_plot, x_plot))

X_plot = convert_x_list_to_array([x_plot2, x_plot2])

X_plot_l = X_plot[:len(x_plot2)]
X_plot_h = X_plot[len(x_plot2):]

## Compute mean predictions and associated variance
lf_mean_lin_mf_model, lf_var_lin_mf_model = lin_mf_model.predict(X_plot_l)
lf_std_lin_mf_model = np.sqrt(lf_var_lin_mf_model)
hf_mean_lin_mf_model, hf_var_lin_mf_model = lin_mf_model.predict(X_plot_h)
hf_std_lin_mf_model = np.sqrt(hf_var_lin_mf_model)

# %%
## Plot the posterior mean and variance (water shielding)
plt.figure(figsize=(10, 6))
#plt.fill_between(x_plot.flatten(), (lf_mean_lin_mf_model - 1.96*lf_std_lin_mf_model).flatten(), (lf_mean_lin_mf_model + 1.96*lf_std_lin_mf_model).flatten(), facecolor='b', alpha=0.3)
plt.fill_between(x_plot.flatten(), (hf_mean_lin_mf_model - 1.96*hf_std_lin_mf_model).flatten(), 
                 (hf_mean_lin_mf_model + 1.96*hf_std_lin_mf_model).flatten(), facecolor='r', alpha=0.3)

#plt.plot(x_plot, lf_mean_lin_mf_model, '--', color='b', label='Predicted Low Fidelity')
plt.plot(x_plot, hf_mean_lin_mf_model, '--', color='r', label='Predicted High Fidelity')
plt.scatter(np.array(x_train_lf_cnp)[:,0], y_train_lf_cnp, color='b', s=5, label='Low Fidelity')
plt.scatter(np.array(x_test_hf_sim)[:,0], y_test_hf_sim, color='r', s=5, label='High Fidelity Test')
plt.xlabel('Water shielding [mm]')
plt.ylabel('CNP value')
plt.legend()
plt.title('Linear multi-fidelity model fit to low and high fidelity Coherent data')
plt.savefig(f'{path_out_mfgp}/water_shielding_fit{FIGURE_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
## Plot the posterior mean and variance (veto thickness)
plt.figure(figsize=(10, 6))
#plt.fill_between(x_plot.flatten(), (lf_mean_lin_mf_model - 1.96*lf_std_lin_mf_model).flatten(), (lf_mean_lin_mf_model + 1.96*lf_std_lin_mf_model).flatten(), facecolor='b', alpha=0.3)
plt.fill_between(x_plot.flatten(), (hf_mean_lin_mf_model - 1.96*hf_std_lin_mf_model).flatten(), 
                 (hf_mean_lin_mf_model + 1.96*hf_std_lin_mf_model).flatten(), facecolor='r', alpha=0.3)

#plt.plot(x_plot, lf_mean_lin_mf_model, '--', color='b', label='Predicted Low Fidelity')
plt.plot(x_plot, hf_mean_lin_mf_model, '--', color='r', label='Predicted High Fidelity')
plt.scatter(np.array(x_train_lf_cnp)[:,1], y_train_lf_cnp, color='b', s=5, label='Low Fidelity')
plt.scatter(np.array(x_test_hf_sim)[:,1], y_test_hf_sim, color='r', s=5, label='High Fidelity Test')
plt.xlabel('Veto thickness [mm]')
plt.ylabel('CNP value')
plt.legend()
plt.title('Linear multi-fidelity model fit to low and high fidelity Coherent data')
plt.savefig(f'{path_out_mfgp}/veto_thickness_fit{FIGURE_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Convert to numpy arrays
x_arr = np.array(x_test_hf_sim)  # your x data
y_arr = np.array(y_test_hf_sim)  # your y values

# Create boolean masks for each group
mask1 = (x_arr[:, 0] == 76.2) & (x_arr[:, 1] == 50.0)
mask2 = (x_arr[:, 0] == 116.2) & (x_arr[:, 1] == 10.0)
mask3 = (x_arr[:, 0] == 26.2) & (x_arr[:, 1] == 100.0)

# Split both x and y data using the same masks
x_group1, y_group1 = x_arr[mask1].tolist(), y_arr[mask1].tolist()
x_group2, y_group2 = x_arr[mask2].tolist(), y_arr[mask2].tolist()
x_group3, y_group3 = x_arr[mask3].tolist(), y_arr[mask3].tolist()

# %%
theta_test = np.array([[76.2,50.0,1],[116.2,10.0,1],[26.2,100.0,1]])

hf_mean, hf_var = lin_mf_model.predict(theta_test)
hf_std = np.sqrt(hf_var)

print(f'(76.2, 50.0) mean: {hf_mean[0]}, var: {hf_var[0]}, std: {hf_std[0]}')
print(f'(116.2, 10.0) mean: {hf_mean[1]}, var: {hf_var[1]}, std: {hf_std[1]}')
print(f'(26.2, 100.0) mean: {hf_mean[2]}, var: {hf_var[2]}, std: {hf_std[2]}')

# %% [markdown]
# ### Plot for (76.2, 50.0) group

# %%
idx_group1 = np.linspace(0, len(x_group1)-1, len(x_group1))
x_plot_g1 = idx_group1[:, None]

vals1 = np.full((len(x_group1), 2), ([76.2,50.0]))
X_plot_g1 = convert_x_list_to_array([vals1, vals1])
X_plot_h_g1 = X_plot_g1[len(vals1):]

## Compute mean predictions and associated variance
hf_mean_lin_mf_model_g1, hf_var_lin_mf_model_g1 = lin_mf_model.predict(X_plot_h_g1)
hf_std_lin_mf_model_g1 = np.sqrt(hf_var_lin_mf_model_g1)

# plotting
plt.figure(figsize=(12, 3))

plt.fill_between(x_plot_g1.flatten(), (hf_mean_lin_mf_model_g1 - 3*hf_std_lin_mf_model_g1).flatten(), (hf_mean_lin_mf_model_g1 + 3*hf_std_lin_mf_model_g1).flatten(), facecolor='r', alpha=0.1, label='$\\pm 3 \\sigma $')
plt.fill_between(x_plot_g1.flatten(), (hf_mean_lin_mf_model_g1 - 2*hf_std_lin_mf_model_g1).flatten(), (hf_mean_lin_mf_model_g1 + 2*hf_std_lin_mf_model_g1).flatten(), facecolor='y', alpha=0.15, label='$\\pm 2 \\sigma $')
plt.fill_between(x_plot_g1.flatten(), (hf_mean_lin_mf_model_g1 - 1*hf_std_lin_mf_model_g1).flatten(), (hf_mean_lin_mf_model_g1 + 1*hf_std_lin_mf_model_g1).flatten(), facecolor='g', alpha=0.2, label='RESuM $\\pm 1 \\sigma $')

plt.scatter(idx_group1, y_group1, color='k', s=5, label='HF Validation Data')
#plt.xlabel('Point index')
plt.ylabel('$y_{raw}$')
plt.title('(76.2, 50.0)')

handles, labels = plt.gca().get_legend_handles_labels()
order = [3,2,1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=4, loc='upper right')

plt.savefig(f'{path_out_mfgp}/validation_group1_76.2_50.0{FIGURE_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
y_data1 = y_group1  # your HF validation y-values
y_center1 = hf_mean_lin_mf_model_g1[0]  # the central RESuM prediction values
sigma1 = hf_std_lin_mf_model_g1[0]  # the uncertainty values (1σ)

# Calculate how many standard deviations each point is from center
deviations1 = np.abs(y_data1 - y_center1) / sigma1

# Count points within each band
within_1sigma = np.sum(deviations1 <= 1)
within_2sigma = np.sum(deviations1 <= 2) 
within_3sigma = np.sum(deviations1 <= 3)

total_points = len(y_data1)
print('Percentages for (76.2, 50.0)')
print(f"±1σ: {within_1sigma}/{total_points} ({100*within_1sigma/total_points:.1f}%)")
print(f"±2σ: {within_2sigma}/{total_points} ({100*within_2sigma/total_points:.1f}%)")
print(f"±3σ: {within_3sigma}/{total_points} ({100*within_3sigma/total_points:.1f}%)")

# Store results for group 1
sigma_results_group1 = {
    'group': '(76.2, 50.0)',
    'total_points': total_points,
    'within_1sigma': within_1sigma,
    'within_2sigma': within_2sigma,
    'within_3sigma': within_3sigma,
    'percent_1sigma': 100*within_1sigma/total_points,
    'percent_2sigma': 100*within_2sigma/total_points,
    'percent_3sigma': 100*within_3sigma/total_points
}

# %% [markdown]
# ### Plot for (116.2, 10.0) group

# %%
idx_group2 = np.linspace(0, len(x_group2)-1, len(x_group2))
x_plot_g2 = idx_group2[:, None]

vals2 = np.full((len(x_group2), 2), ([116.2,10.0]))
X_plot_g2 = convert_x_list_to_array([vals2, vals2])
X_plot_h_g2 = X_plot_g2[len(vals2):]

## Compute mean predictions and associated variance
hf_mean_lin_mf_model_g2, hf_var_lin_mf_model_g2 = lin_mf_model.predict(X_plot_h_g2)
hf_std_lin_mf_model_g2 = np.sqrt(hf_var_lin_mf_model_g2)

# plotting
plt.figure(figsize=(12, 3))

plt.fill_between(x_plot_g2.flatten(), (hf_mean_lin_mf_model_g2 - 3*hf_std_lin_mf_model_g2).flatten(), (hf_mean_lin_mf_model_g2 + 3*hf_std_lin_mf_model_g2).flatten(), facecolor='r', alpha=0.1, label='$\\pm 3 \\sigma $')
plt.fill_between(x_plot_g2.flatten(), (hf_mean_lin_mf_model_g2 - 2*hf_std_lin_mf_model_g2).flatten(), (hf_mean_lin_mf_model_g2 + 2*hf_std_lin_mf_model_g2).flatten(), facecolor='y', alpha=0.15, label='$\\pm 2 \\sigma $')
plt.fill_between(x_plot_g2.flatten(), (hf_mean_lin_mf_model_g2 - 1*hf_std_lin_mf_model_g2).flatten(), (hf_mean_lin_mf_model_g2 + 1*hf_std_lin_mf_model_g2).flatten(), facecolor='g', alpha=0.2, label='RESuM $\\pm 1 \\sigma $')

plt.scatter(idx_group2, y_group2, color='k', s=5, label='HF Validation Data')
plt.ylabel('$y_{raw}$')
plt.title('(116.2, 10.0)')

handles, labels = plt.gca().get_legend_handles_labels()
order = [3,2,1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=4, loc='upper right')

plt.savefig(f'{path_out_mfgp}/validation_group2_116.2_10.0{FIGURE_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
y_data2 = y_group2  # your HF validation y-values
y_center2 = hf_mean_lin_mf_model_g2[0]  # the central RESuM prediction values
sigma2 = hf_std_lin_mf_model_g2[0]  # the uncertainty values (1σ)

# Calculate how many standard deviations each point is from center
deviations2 = np.abs(y_data2 - y_center2) / sigma2

# Count points within each band
within_1sigma = np.sum(deviations2 <= 1)
within_2sigma = np.sum(deviations2 <= 2) 
within_3sigma = np.sum(deviations2 <= 3)

total_points = len(y_data2)
print('Percentages for (116.2, 10.0)')
print(f"±1σ: {within_1sigma}/{total_points} ({100*within_1sigma/total_points:.1f}%)")
print(f"±2σ: {within_2sigma}/{total_points} ({100*within_2sigma/total_points:.1f}%)")
print(f"±3σ: {within_3sigma}/{total_points} ({100*within_3sigma/total_points:.1f}%)")

# Store results for group 2
sigma_results_group2 = {
    'group': '(116.2, 10.0)',
    'total_points': total_points,
    'within_1sigma': within_1sigma,
    'within_2sigma': within_2sigma,
    'within_3sigma': within_3sigma,
    'percent_1sigma': 100*within_1sigma/total_points,
    'percent_2sigma': 100*within_2sigma/total_points,
    'percent_3sigma': 100*within_3sigma/total_points
}

# %% [markdown]
# ### Plot for (26.2, 100.0) group

# %%
idx_group3 = np.linspace(0, len(x_group3)-1, len(x_group3))
x_plot_g3 = idx_group3[:, None]

vals3 = np.full((len(x_group3), 2), ([26.2,100.0]))
X_plot_g3 = convert_x_list_to_array([vals3, vals3])
X_plot_h_g3 = X_plot_g3[len(vals3):]

## Compute mean predictions and associated variance
hf_mean_lin_mf_model_g3, hf_var_lin_mf_model_g3 = lin_mf_model.predict(X_plot_h_g3)
hf_std_lin_mf_model_g3 = np.sqrt(hf_var_lin_mf_model_g3)

# plotting
plt.figure(figsize=(12, 3))

plt.fill_between(x_plot_g3.flatten(), (hf_mean_lin_mf_model_g3 - 3*hf_std_lin_mf_model_g3).flatten(), (hf_mean_lin_mf_model_g3 + 3*hf_std_lin_mf_model_g3).flatten(), facecolor='r', alpha=0.1, label='$\\pm 3 \\sigma $')
plt.fill_between(x_plot_g3.flatten(), (hf_mean_lin_mf_model_g3 - 2*hf_std_lin_mf_model_g3).flatten(), (hf_mean_lin_mf_model_g3 + 2*hf_std_lin_mf_model_g3).flatten(), facecolor='y', alpha=0.15, label='$\\pm 2 \\sigma $')
plt.fill_between(x_plot_g3.flatten(), (hf_mean_lin_mf_model_g3 - 1*hf_std_lin_mf_model_g3).flatten(), (hf_mean_lin_mf_model_g3 + 1*hf_std_lin_mf_model_g3).flatten(), facecolor='g', alpha=0.2, label='RESuM $\\pm 1 \\sigma $')

plt.scatter(idx_group3, y_group3, color='k', s=5, label='HF Validation Data')
plt.ylabel('$y_{raw}$')
plt.title('(26.2, 100.0)')

handles, labels = plt.gca().get_legend_handles_labels()
order = [3,2,1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=4, loc='lower right')

plt.savefig(f'{path_out_mfgp}/validation_group3_26.2_100.0{FIGURE_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
y_data3 = y_group3  # your HF validation y-values
y_center3 = hf_mean_lin_mf_model_g3[0]  # the central RESuM prediction values
sigma3 = hf_std_lin_mf_model_g3[0]  # the uncertainty values (1σ)

# Calculate how many standard deviations each point is from center
deviations3 = np.abs(y_data3 - y_center3) / sigma3

# Count points within each band
within_1sigma = np.sum(deviations3 <= 1)
within_2sigma = np.sum(deviations3 <= 2) 
within_3sigma = np.sum(deviations3 <= 3)

total_points = len(y_data3)
print('Percentages for (26.2, 100.0)')
print(f"±1σ: {within_1sigma}/{total_points} ({100*within_1sigma/total_points:.1f}%)")
print(f"±2σ: {within_2sigma}/{total_points} ({100*within_2sigma/total_points:.1f}%)")
print(f"±3σ: {within_3sigma}/{total_points} ({100*within_3sigma/total_points:.1f}%)")

# Store results for group 3
sigma_results_group3 = {
    'group': '(26.2, 100.0)',
    'total_points': total_points,
    'within_1sigma': within_1sigma,
    'within_2sigma': within_2sigma,
    'within_3sigma': within_3sigma,
    'percent_1sigma': 100*within_1sigma/total_points,
    'percent_2sigma': 100*within_2sigma/total_points,
    'percent_3sigma': 100*within_3sigma/total_points
}

# %%
# Save sigma coverage results to text file
sigma_results = [sigma_results_group1, sigma_results_group2, sigma_results_group3]

with open(f'{path_out_mfgp}/sigma_coverage_results{FIGURE_SUFFIX}.txt', 'w') as f:
    f.write(f"Sigma Coverage Analysis Results - {FIGURE_SUFFIX}\n")
    f.write("="*60 + "\n\n")
    
    for result in sigma_results:
        f.write(f"Parameter Group: {result['group']}\n")
        f.write(f"Total Points: {result['total_points']}\n")
        f.write(f"±1σ: {result['within_1sigma']}/{result['total_points']} ({result['percent_1sigma']:.1f}%)\n")
        f.write(f"±2σ: {result['within_2sigma']}/{result['total_points']} ({result['percent_2sigma']:.1f}%)\n")
        f.write(f"±3σ: {result['within_3sigma']}/{result['total_points']} ({result['percent_3sigma']:.1f}%)\n")
        f.write("-" * 40 + "\n\n")
    
    # Summary statistics
    f.write("Summary Statistics:\n")
    f.write("-" * 20 + "\n")
    avg_1sigma = np.mean([r['percent_1sigma'] for r in sigma_results])
    avg_2sigma = np.mean([r['percent_2sigma'] for r in sigma_results])
    avg_3sigma = np.mean([r['percent_3sigma'] for r in sigma_results])
    
    f.write(f"Average ±1σ coverage: {avg_1sigma:.1f}%\n")
    f.write(f"Average ±2σ coverage: {avg_2sigma:.1f}%\n")
    f.write(f"Average ±3σ coverage: {avg_3sigma:.1f}%\n")
    
    total_all_points = sum([r['total_points'] for r in sigma_results])
    total_1sigma = sum([r['within_1sigma'] for r in sigma_results])
    total_2sigma = sum([r['within_2sigma'] for r in sigma_results])
    total_3sigma = sum([r['within_3sigma'] for r in sigma_results])
    
    f.write(f"\nOverall (all groups combined):\n")
    f.write(f"±1σ: {total_1sigma}/{total_all_points} ({100*total_1sigma/total_all_points:.1f}%)\n")
    f.write(f"±2σ: {total_2sigma}/{total_all_points} ({100*total_2sigma/total_all_points:.1f}%)\n")
    f.write(f"±3σ: {total_3sigma}/{total_all_points} ({100*total_3sigma/total_all_points:.1f}%)\n")

print(f"Sigma coverage results saved to {path_out_mfgp}/sigma_coverage_results{FIGURE_SUFFIX}.txt")

# %% [markdown]
# ### Contour Heat Map Visualization
# 
# Create a contour heat map showing the predicted y_raw values across the parameter space.

# %%
def draw_yraw_contour(mf_model, x_labels, param_x_idx, param_y_idx, x_fixed, grid_steps=40, levels=20, cmap="viridis"):
    """Draw a filled contour plot of the high-fidelity (y_raw) prediction as function of two parameters.

    Parameters
    ----------
    mf_model : GPyMultiOutputWrapper
        The multi-fidelity model.
    x_labels : list
        List of parameter names.
    param_x_idx, param_y_idx : int
        Indices of the parameters to be shown on the horizontal and vertical axes, respectively.
    x_fixed : array-like
        Fixed values for all parameters (will be modified for the two varying parameters).
    grid_steps : int, optional
        Number of grid points along each axis.
    levels : int, optional
        Number of contour levels.
    cmap : str, optional
        Colormap for the filled contours.
    """
    
    # Get parameter ranges from the data
    param_x_min, param_x_max = data[x_labels[param_x_idx]].min(), data[x_labels[param_x_idx]].max()
    param_y_min, param_y_max = data[x_labels[param_y_idx]].min(), data[x_labels[param_y_idx]].max()
    
    # Build grid for the two selected parameters
    x_vals = np.linspace(param_x_min, param_x_max, grid_steps)
    y_vals = np.linspace(param_y_min, param_y_max, grid_steps)
    Xg, Yg = np.meshgrid(x_vals, y_vals)

    # Prepare points to evaluate
    points = []
    for y in y_vals:
        for x in x_vals:
            x_vec = np.array(x_fixed).copy()
            x_vec[param_x_idx] = x
            x_vec[param_y_idx] = y
            points.append(x_vec)

    points = np.array(points)
    
    # For high-fidelity prediction, we need to add fidelity indicator
    # Add fidelity column (1 for high fidelity)
    fidelity_col = np.ones((len(points), 1))
    points_with_fidelity = np.hstack([points, fidelity_col])
    
    # Evaluate model at high fidelity
    mean_pred, _ = mf_model.predict(points_with_fidelity)
    Z = mean_pred.reshape(grid_steps, grid_steps)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    contour = ax.contourf(Xg, Yg, Z, levels=levels, cmap=cmap)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r"Predicted $y_{raw}$", fontsize=12)

    ax.set_xlabel(x_labels[param_x_idx], fontsize=12)
    ax.set_ylabel(x_labels[param_y_idx], fontsize=12)
    ax.set_title(r"Predicted $y_{raw}$ Contour Map", fontsize=14)

    # Add contour lines for better readability
    contour_lines = ax.contour(Xg, Yg, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Overlay training data points
    ax.scatter(np.array(x_train_hf_sim)[:, param_x_idx], 
               np.array(x_train_hf_sim)[:, param_y_idx], 
               c='red', s=100, marker='o', edgecolors='black', linewidth=2, 
               label='HF Training Data', zorder=5)
    
    # Overlay low-fidelity training data points
    ax.scatter(np.array(x_train_lf_cnp)[:, param_x_idx], 
               np.array(x_train_lf_cnp)[:, param_y_idx], 
               c='blue', s=30, marker='s', alpha=0.6, 
               label='LF Training Data', zorder=4)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return fig

# %%
# Create contour heat map for water_shielding_mm (x-axis) vs veto_thickness_mm (y-axis)
# Define fixed parameter values (using mean values from the training data)
x_fixed = [np.mean(np.array(x_train_lf_cnp)[:, i]) for i in range(len(x_labels))]

# Create the contour plot
# x_labels[0] should be water_shielding_mm, x_labels[1] should be veto_thickness_mm
param_x_idx = 1  # water_shielding_mm
param_y_idx = 0  # veto_thickness_mm

fig_contour = draw_yraw_contour(lin_mf_model, x_labels, param_x_idx, param_y_idx, x_fixed, 
                               grid_steps=50, levels=25, cmap="viridis")
plt.show()

# %%
# Create a more detailed contour plot with uncertainty visualization
def draw_yraw_contour_with_uncertainty(mf_model, x_labels, param_x_idx, param_y_idx, x_fixed, grid_steps=40, levels=20, cmap="viridis"):
    """Draw contour plots showing both mean prediction and uncertainty."""
    
    # Get parameter ranges from the data
    param_x_min, param_x_max = data[x_labels[param_x_idx]].min(), data[x_labels[param_x_idx]].max()
    param_y_min, param_y_max = data[x_labels[param_y_idx]].min(), data[x_labels[param_y_idx]].max()
    
    # Build grid for the two selected parameters
    x_vals = np.linspace(param_x_min, param_x_max, grid_steps)
    y_vals = np.linspace(param_y_min, param_y_max, grid_steps)
    Xg, Yg = np.meshgrid(x_vals, y_vals)

    # Prepare points to evaluate
    points = []
    for y in y_vals:
        for x in x_vals:
            x_vec = np.array(x_fixed).copy()
            x_vec[param_x_idx] = x
            x_vec[param_y_idx] = y
            points.append(x_vec)

    points = np.array(points)
    
    # For high-fidelity prediction, we need to add fidelity indicator
    fidelity_col = np.ones((len(points), 1))
    points_with_fidelity = np.hstack([points, fidelity_col])
    
    # Evaluate model at high fidelity
    mean_pred, var_pred = mf_model.predict(points_with_fidelity)
    std_pred = np.sqrt(var_pred)
    
    Z_mean = mean_pred.reshape(grid_steps, grid_steps)
    Z_std = std_pred.reshape(grid_steps, grid_steps)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    
    # Plot 1: Mean prediction
    contour1 = ax1.contourf(Xg, Yg, Z_mean, levels=levels, cmap=cmap)
    cbar1 = fig.colorbar(contour1, ax=ax1)
    cbar1.set_label(r"Predicted $y_{raw}$ (mean)", fontsize=12)
    
    ax1.set_xlabel(x_labels[param_x_idx], fontsize=12)
    ax1.set_ylabel(x_labels[param_y_idx], fontsize=12)
    ax1.set_title(r"Mean Prediction", fontsize=14)
    
    # Add contour lines and training data
    ax1.contour(Xg, Yg, Z_mean, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    ax1.scatter(np.array(x_train_hf_sim)[:, param_x_idx], 
               np.array(x_train_hf_sim)[:, param_y_idx], 
               c='red', s=100, marker='o', edgecolors='black', linewidth=2, 
               label='HF Training Data', zorder=5)
    ax1.scatter(np.array(x_train_lf_cnp)[:, param_x_idx], 
               np.array(x_train_lf_cnp)[:, param_y_idx], 
               c='blue', s=30, marker='s', alpha=0.6, 
               label='LF Training Data', zorder=4)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty (standard deviation)
    contour2 = ax2.contourf(Xg, Yg, Z_std, levels=levels, cmap='Reds')
    cbar2 = fig.colorbar(contour2, ax=ax2)
    cbar2.set_label(r"Uncertainty ($\sigma$)", fontsize=12)
    
    ax2.set_xlabel(x_labels[param_x_idx], fontsize=12)
    ax2.set_ylabel(x_labels[param_y_idx], fontsize=12)
    ax2.set_title(r"Prediction Uncertainty", fontsize=14)
    
    # Add contour lines and training data
    ax2.contour(Xg, Yg, Z_std, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    ax2.scatter(np.array(x_train_hf_sim)[:, param_x_idx], 
               np.array(x_train_hf_sim)[:, param_y_idx], 
               c='red', s=100, marker='o', edgecolors='black', linewidth=2, 
               label='HF Training Data', zorder=5)
    ax2.scatter(np.array(x_train_lf_cnp)[:, param_x_idx], 
               np.array(x_train_lf_cnp)[:, param_y_idx], 
               c='blue', s=30, marker='s', alpha=0.6, 
               label='LF Training Data', zorder=4)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    return fig

# Create the enhanced contour plot with uncertainty
fig_contour_detailed = draw_yraw_contour_with_uncertainty(lin_mf_model, x_labels, param_x_idx, param_y_idx, x_fixed, 
                                                         grid_steps=50, levels=25, cmap="viridis")
plt.show()

# %%
x_train_hf_sim

# %%
# Save the contour plots
fig_contour.savefig(f'{path_out_mfgp}/yraw_contour_map{FIGURE_SUFFIX}.png', dpi=300, bbox_inches='tight')
fig_contour_detailed.savefig(f'{path_out_mfgp}/yraw_contour_with_uncertainty{FIGURE_SUFFIX}.png', dpi=300, bbox_inches='tight') 

print(f"Contour plots saved to {path_out_mfgp}/")
print(f"- yraw_contour_map{FIGURE_SUFFIX}.png")
print(f"- yraw_contour_with_uncertainty{FIGURE_SUFFIX}.png")

# %%
# Summary of all saved files
print(f"\n" + "="*60)
print(f"SUMMARY: All files saved with suffix '{FIGURE_SUFFIX}'")
print(f"="*60)
print(f"Output directory: {path_out_mfgp}/")
print(f"\nFigures saved:")
print(f"- water_shielding_fit{FIGURE_SUFFIX}.png")
print(f"- veto_thickness_fit{FIGURE_SUFFIX}.png")
print(f"- validation_group1_76.2_50.0{FIGURE_SUFFIX}.png")
print(f"- validation_group2_116.2_10.0{FIGURE_SUFFIX}.png") 
print(f"- validation_group3_26.2_100.0{FIGURE_SUFFIX}.png")
print(f"- yraw_contour_map{FIGURE_SUFFIX}.png")
print(f"- yraw_contour_with_uncertainty{FIGURE_SUFFIX}.png")
print(f"\nData files saved:")
print(f"- sigma_coverage_results{FIGURE_SUFFIX}.txt")
print(f"="*60)


