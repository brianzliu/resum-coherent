# COHERENT Detector Design Optimization with RESuM

Efficient optimization of COHERENT detector design parameters using the Rare Event Surrogate Model (RESuM).

**Paper**: [Efficient optimization of COHERENT detector design parameters with the Rare Event Surrogate Model (RESUM)](https://openreview.net/pdf?id=m9BSQBkE0e)

## Overview

This repository applies RESuM (Rare Event Surrogate Model) to optimize the COH-Ar-750 detector design for the COHERENT collaboration.

### Background
Coherent elastic neutrino-nucleus scattering (CEνNS) is a weak neutral-current process in which a neutrino scatters off a nucleus as a whole. Following the initial observation by COHERENT in 2017, the next-generation COH-Ar-750 detector is being developed to measure CEνNS with percent-level precision and probe for physics beyond the Standard Model.

### Design Challenge
A primary challenge is mitigating neutron backgrounds from the Spallation Neutron Source (SNS) at Oak Ridge National Laboratory (ORNL). Neutron-induced nuclear recoils produce signals nearly indistinguishable from CEνNS, requiring extensive Monte Carlo event simulations to optimize shielding arrangements. The detector's veto system combines:
- **Passive shielding**: Lead and water blocks
- **Active shielding**: Plastic scintillator panels

### RESuM Solution
Because neutrons have a low probability of depositing energy in the active liquid argon volume, optimizing the shielding design is a rare event problem requiring expensive simulations. RESuM addresses this by:
- Using a Conditional Neural Process (CNP) to incorporate physics priors
- Employing Multi-Fidelity Gaussian Process (MFGP) modeling to blend low- and high-fidelity simulations
- Adaptively sampling to guide expensive simulations only where needed

### Design Parameters Optimized
- **Water shielding thickness** (mm): Passive shielding material
- **Veto panel thickness** (mm): Active scintillator detector thickness

### Results
Experimental results suggest thicker water shielding and thinner veto panels increase neutron rejection efficiency. RESuM achieved:
- Correlation coefficient: **r = 0.880**
- Well-calibrated uncertainties
- Significant reduction in required high-fidelity simulations

This demonstrates RESuM's potential to accelerate design optimization for a broad range of rare event search experiments.

## Installation and Setup

### Step 1: Create Conda Environment

Create a new conda environment using the provided environment file:
```bash
conda env create -f coherent_environment.yml
conda activate coherent
```

### Step 2: Download Dataset

Download the COHERENT dataset from Zenodo:
- **Dataset URL**: [https://zenodo.org/records/17299286](https://zenodo.org/records/17299286)
- **File**: `coherent_dataset.tar.gz` (496.5 MB)

### Step 3: Place Dataset

Move the downloaded tarball to the data directory:
```bash
mv coherent_dataset.tar.gz src/coherent/in/data/
```

### Step 4: Preprocess Data

Run the preprocessing script to extract and organize the dataset:
```bash
python src/coherent/in/data/preprocessing.py
```

This script will:
- Extract the tarball
- Stratify data into training and validation sets
- Create output directories (`out/cnp`, `out/mfgp`, `out/pce`)
- Generate corresponding HDF5 files for all CSV data

### Step 5: Run Analysis Pipeline

Execute the Jupyter notebooks in the following order:

1. **Train Conditional Neural Process (CNP)**:
   ```bash
   jupyter notebook src/run_cnp/conditional_neural_process_training_coherent.ipynb
   ```

2. **Generate CNP Predictions**:
   
   Run the prediction notebook **twice** with different settings in `settings.yaml`:
   
   a. First, set `path_to_files_predict: ["../coherent/in/data/training/lf", "../coherent/in/data/training/hf/"]` to generate training predictions:
   ```bash
   jupyter notebook src/run_cnp/conditional_neural_process_predict_coherent.ipynb
   ```
   This creates `cnp_{version}_output.csv` and `cnp_{version}_output.png`
   
   b. Then, set `path_to_files_predict: ["../coherent/in/data/validation/lf"]` to generate validation predictions:
   ```bash
   jupyter notebook src/run_cnp/conditional_neural_process_predict_coherent.ipynb
   ```
   This creates `cnp_{version}_output_validation.csv` and `cnp_{version}_output_validation.png`

3. **Run Multi-Fidelity Gaussian Process (MFGP)**:
   ```bash
   jupyter notebook src/run_mfgp/mfgp_coherent.ipynb
   ```

All outputs will be saved in the `src/coherent/out/` directory.
