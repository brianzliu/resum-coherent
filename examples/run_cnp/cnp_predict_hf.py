# %% [markdown]
# # Conditional Neural Processes (CNP) for Coherent.
# [Conditional Neural Processes](https://arxiv.org/pdf/1807.01613.pdf) (CNPs) were
# introduced as a continuation of
# [Generative Query Networks](https://deepmind.com/blog/neural-scene-representation-and-rendering/)
# (GQN) to extend its training regime to tasks beyond scene rendering, e.g. to
# regression and classification.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import datetime
import numpy as np
import pandas as pd
import os
import gc  # Add garbage collection
from resum.utilities import plotting_utils_cnp as plotting
from resum.utilities import utilities as utils
try:
    from resum.conditional_neural_process import DeterministicModel
    from resum.conditional_neural_process import DataGeneration
except Exception as e:
    print(f"Error occurred: {e}. Retrying import...")
    from resum.conditional_neural_process import DeterministicModel
    from resum.conditional_neural_process import DataGeneration
from torch.utils.tensorboard import SummaryWriter
import csv
import yaml

# %%
with open("../coherent/settings_newdata.yaml", "r") as f:
    config_file = yaml.safe_load(f)

PLOT_AFTER = int(config_file["cnp_settings"]["plot_after"])
torch.manual_seed(0)
FILES_PER_BATCH = config_file["cnp_settings"]["files_per_batch_predict"]
target_range = config_file["simulation_settings"]["target_range"]
is_binary = target_range[0] >= 0 and target_range[1] <= 1

path_out  = config_file["path_settings"]["path_out_cnp"]
version   = config_file["path_settings"]["version"]
iteration = config_file["path_settings"]["iteration"]
fidelity  = config_file["path_settings"]["fidelity"]

# %%
x_size, y_size = utils.get_feature_and_label_size(config_file)
theta_size=len(config_file["simulation_settings"]["theta_headers"])

# %%
d_x, d_in, representation_size, d_out = x_size , x_size+y_size, 32, y_size+1
encoder_sizes = [d_in, 32, 64, 128, 128, 128, 64, 48, representation_size]
decoder_sizes = [representation_size + d_x, 32, 64, 128, 128, 128, 64, 48, d_out]

model = DeterministicModel(encoder_sizes,decoder_sizes)
model.load_state_dict(torch.load(f'{path_out}/cnp_{version}_model_20epochs.pth'))
model.eval()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# %%
bce = nn.BCELoss()

# create a PdfPages object
test_idx=0
it_batch = 0

with open(f'{path_out}/cnp_{version}_output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    headers = ['iteration','fidelity','n_samples'] + [*config_file["simulation_settings"]["theta_headers"]] + ['y_cnp', 'y_cnp_err', 'y_raw', 'log_prop','bce']
    writer.writerow(headers)

for p, path in enumerate(config_file["path_settings"]["path_to_files_predict"]):
    USE_DATA_AUGMENTATION = False
    # load data:

    dataset_predict = DataGeneration(mode = "testing", config_file=config_file, 
                                   path_to_files=path, 
                                   use_data_augmentation=USE_DATA_AUGMENTATION, 
                                   batch_size=config_file["cnp_settings"]["batch_size_predict"][p], 
                                   files_per_batch=FILES_PER_BATCH)
    dataset_predict.set_loader()
    dataloader_predict = dataset_predict.dataloader

    # Shuffle the files in the dataset for randomized prediction order
    dataloader_predict.dataset.shuffle_files()

    for b, batch in enumerate(dataloader_predict):
        # Safety check: Skip empty batches
        if batch.numel() == 0:
            print(f"Warning: Skipping empty batch at iteration {b}")
            continue
        
        # Use torch.no_grad() to disable gradient computation and save memory
        with torch.no_grad():
            # Move batch to device
            batch = batch.to(device)
            
            batch_formated = dataset_predict.format_batch_for_cnp(batch, config_file["cnp_settings"]["context_is_subset"])
            
            # Move formatted batch components to device
            # Handle nested tuple structure: ((context_x, context_y), target_x)
            (context_x, context_y), target_x = batch_formated.query
            query_on_device = ((context_x.to(device), context_y.to(device)), target_x.to(device))
            target_y_on_device = batch_formated.target_y.to(device)
            
            # Create new namedtuple with tensors on device
            from resum.conditional_neural_process.data_generator import CNPRegressionDescription
            batch_formated = CNPRegressionDescription(query=query_on_device, target_y=target_y_on_device)
            
            # Get the predicted mean and variance at the target points for the testing set
            log_prob, mu, sigma = model(batch_formated.query, batch_formated.target_y, is_binary)
            
            # Define the loss
            loss = -log_prob.mean()
            
            if is_binary:
                loss_bce = bce(mu, batch_formated.target_y)
            else:
                loss_bce = torch.tensor(-1.0)

            # Move results to CPU and convert to numpy immediately
            mu_cpu = mu[0].cpu().detach().numpy()
            sigma_cpu = sigma[0].cpu().detach().numpy()
            theta_cpu = batch_formated.query[1][0].cpu().detach().numpy()
            target_y_cpu = batch_formated.target_y.cpu().detach().numpy()
            loss_cpu = loss.cpu().item()
            loss_bce_cpu = loss_bce.cpu().item()
            
            # Clear GPU memory
            del batch, batch_formated, log_prob, mu, sigma, loss, loss_bce
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Batch: {it_batch}")
        row = np.array([iteration[p]] + [fidelity[p]] + [len(mu_cpu)] + theta_cpu[0,:theta_size].tolist() + [np.mean(mu_cpu), np.sqrt(np.sum(sigma_cpu**2)) / len(sigma_cpu), np.mean(target_y_cpu), loss_cpu, loss_bce_cpu])
        # Reshape to 2D array (one row)
        row = row.reshape(1, -1)

        # Write the row to the CSV file with 5 decimal places for each number
        with open(f'{path_out}/cnp_{version}_output.csv', mode='a', newline='') as file:
            np.savetxt(file, row, delimiter=",", fmt="%.5f")

        # if it_batch % PLOT_AFTER == 0:
        #     mu_predict = mu_cpu
        #     loss_predict = loss_cpu
        #     print('{} Iteration: {}/{}, train loss: {:.4f} (vs BCE {:.4f})'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), p, it_batch, loss_cpu, loss_bce_cpu))
        #     if y_size == 1:
        #         fig = plotting.plot(mu_cpu, target_y_cpu[0], f'{loss_cpu:.2f}', mu_predict, target_y_cpu[0], f'{loss_predict:.2f}', target_range, it_batch)
        #     else:
        #         for k in range(y_size):
        #             fig = plotting.plot(mu_cpu[:,k], target_y_cpu[0][:,k], f'{loss_cpu:.2f}', mu_predict[:,k], target_y_cpu[0][:,k], f'{loss_predict:.2f}', target_range, it_batch)
        
        # Force garbage collection every 100 batches for high fidelity data
        if p == 1 and it_batch % 100 == 0:  # p=1 is high fidelity
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"Memory cleanup at batch {it_batch}")
        
        it_batch += 1

# %%
# Generate marginalized plots for all parameters
try:
    # Temporarily create a copy with the expected filename
    import shutil
    expected_file = f'{path_out}/cnp_{version}_output.csv'
    actual_file = f'{path_out}/cnp_{version}_coherent_output_valid_20epochs.csv'
    
    if os.path.exists(actual_file) and not os.path.exists(expected_file):
        shutil.copy2(actual_file, expected_file)
    
    fig = plotting.get_marginialized_all(config_file=config_file)
    fig.savefig(f'{path_out}/cnp_{version}_output_valid_20epochs.png', dpi=300, bbox_inches='tight')
    print(f"Marginalized plots saved to: {path_out}/cnp_{version}_output_valid_20epochs.png")
    
    # Clean up the temporary file
    if os.path.exists(expected_file) and os.path.exists(actual_file):
        os.remove(expected_file)
        
except Exception as e:
    print(f"Error generating marginalized plots: {e}")
    import traceback
    traceback.print_exc()

# %%
import os

# Rename the coherent output file to include 20epochs_hf suffix
old_file = '/home/bliu4/resum-coherent2/examples/coherent/out/cnp/newdata/cnp_v105.0_coherent_output.csv'
new_file = '/home/bliu4/resum-coherent2/examples/coherent/out/cnp/newdata/cnp_v105.0_coherent_output_valid_20epochs.csv'

if os.path.exists(old_file):
    os.rename(old_file, new_file)
    print(f"File renamed successfully:")
    print(f"From: {old_file}")
    print(f"To: {new_file}")
else:
    print(f"Source file not found: {old_file}")
    # Check if the target file already exists
    if os.path.exists(new_file):
        print(f"Target file already exists: {new_file}")
    else:
        print("Neither source nor target file exists.")