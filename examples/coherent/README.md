# Coherent Neutrino Scattering Detector Design with RESuM

This directory contains the setup for applying RESuM (Rare Event Surrogate Model) to optimize coherent neutrino scattering detector designs.

## Problem Setup

### Rare Event Design (RED) Problem
We want to optimize detector design parameters (θ) to minimize the probability of background events contaminating the detector while maximizing signal detection efficiency.

### RESuM Methodology

#### Design Parameters (θ) - What we optimize:
Based on actual experimental configurations:
- `water_shielding_mm`: Water shielding thickness (16.2-116.2 mm)
- `veto_thickness_mm`: Veto detector thickness (10-110 mm)

These represent the actual experimental variables that distinguish the different detector configurations (baseline, Veto1, Veto3, etc.).

#### Event-specific Parameters (φ) - Physics of each event:
- `fGenX`, `fGenY`, `fGenZ`: Generation coordinates
- `fEventEnergy`: Event energy
- `fMomentumX`, `fMomentumY`, `fMomentumZ`: Momentum components
- `fEDepNR`: Nuclear recoil energy deposition
- `fEDepVeto`: Veto energy deposition

#### Target Variable:
- `target_active`: Binary outcome (1 = veto_active==0 target_active==1 (detector hit coincident with veto miss))

### Current Data Files
The simulation files (`g4coherent_*.csv`) contain event-level physics data for different veto configurations. Each row represents one simulated particle event with its physics parameters and outcome.

#### Training Data
- **High Fidelity (HF)**: Full physics simulations (Veto1, Veto26, Veto76)
- **Low Fidelity (LF)**: Simplified physics simulations (Veto11, Veto21, Veto31, Veto41, Veto51, Veto61, Veto71, Veto81, Veto91)

#### Validation Data
- **Low Fidelity (LF)**: Extensive validation set with 100+ veto configurations (Veto2-Veto100) for comprehensive testing

### Required Data Processing

To properly implement RESuM, the existing simulation data needs to be processed to:

1. **Map Design Configurations**: Map each veto configuration to its actual design parameter values (water shielding and veto thickness)
2. **Aggregate Event Data**: For each design configuration, calculate the design metric `y = m/N` where:
   - `m` = number of signal events (veto_active = 1)
   - `N` = total number of events simulated
3. **Extract Event Features**: Use the physics parameters as event-specific features (φ)

### Pipeline Steps

1. **CNP Training**: Train on event-level data to learn the relationship between design parameters θ, event features φ, and binary outcomes
2. **Design Metric Calculation**: Compute `y_CNP` (CNP predictions) and `y_Raw` (actual rates) for different designs
3. **MFGP Training**: Use multi-fidelity GP to combine LF and HF design metrics
4. **Optimization**: Use active learning to find optimal design parameters

## File Organization

```
coherent/
├── in/
│   ├── data/
│   │   ├── processed/           # Original processed data
│   │   │   ├── lf/              # Low fidelity (Veto4-Veto11)
│   │   │   └── hf/              # High fidelity (baseline, Veto1, Veto3)
│   │   ├── processed_newdata/   # New dataset structure
│   │   │   ├── training/
│   │   │   │   ├── hf/          # HF training (Veto1, Veto26, Veto76)
│   │   │   │   └── lf/          # LF training (Veto11, Veto21, etc.)
│   │   │   └── validation/
│   │   │       └── lf/          # LF validation (100+ configurations)
│   │   ├── preprocessing.py     # Data preprocessing utilities
│   │   └── run_comparison.py    # Data comparison tools
│   └── mfgp/                    # Multi-fidelity GP inputs
├── out/
│   ├── cnp/                     # CNP model outputs
│   │   ├── newdata/             # Results for new dataset
│   │   └── *.pth, *.csv, *.png  # Model files and results
│   ├── mfgp/                    # MFGP optimization results
│   │   ├── newdata/             # New dataset MFGP results
│   │   └── *.png                # Visualization outputs
│   └── pce/                     # Polynomial Chaos Expansion outputs
│       └── newdata/             # New dataset PCE results
├── simulations/                 # Original simulation files
├── settings.yaml                # Original configuration
├── settings_newdata.yaml        # New dataset configuration
├── inequalities.py              # Design constraints
└── README.md                   # This file
```

## Key Differences from LEGEND Neutron Moderator

- **Application Domain**: Coherent neutrino scattering vs neutron background reduction
- **Design Parameters**: Water shielding + veto thickness vs moderator geometry (radius, thickness, panels, angle, length)
- **Physics**: Neutrino-nucleus interactions vs neutron moderation
- **Constraints**: Shielding effectiveness vs geometric feasibility

## Workflow and Usage

### 1. Data Preprocessing
- Convert CSV files to HDF5 format for efficient training
- Data is organized into training/validation splits with HF/LF fidelity levels

### 2. CNP Training and Prediction
- Configure `settings_newdata.yaml` for your specific run:
  - For training: Set paths to `processed_newdata/training/lf/` and `processed_newdata/training/hf/`
  - For validation: Set paths to `processed_newdata/validation/lf/`
- Run CNP training to learn event-level physics relationships
- Generate predictions for design optimization

### 3. MFGP Analysis
- Use `mfgp_analyzer.py` (already implemented in `../run_mfgp/sonata_mfgp.ipynb` at the bottom) for comprehensive analysis
- Automated pipeline includes:
  - Uncertainty band visualization across theta values
  - Coverage statistics and validation
  - Enhanced contour plots showing mean predictions and uncertainty
  - Prediction vs true value comparisons

### 4. Results and Visualization
- All outputs saved to respective directories in `out/`
- CNP models and predictions in `out/cnp/newdata/`
- MFGP results and plots in `out/mfgp/newdata/`
- Automated plotting generates publication-ready figures

## References

- RESuM Paper: "Rare Event Surrogate Model for Physics Detector Design" (ICLR 2025)
- LEGEND Collaboration neutron moderator optimization example
- Coherent neutrino scattering simulation framework 

## how to run (for sonata)
- the filetree for data is also different, i have a folder "coherent/in/data/processed_newdata"
- inside we have "processed_newdata/training" and "processed_newdata/validation"
- in training we have "processed_newdata/training/hf" (veto1, veto2, veto7) and "processed_newdata/training/lf" (veto11, veto21, veto31, veto41, veto51, veto61, veto71, veo81, veto91)
- in validation we have "processed_newdata/validation/lf" with the rest of the files
- to convert to h5 i honestly just take the original csv-to-h5 conversion code and tell Cursor to adapt it to the csvs in the filetree

- depending on whether you run cnp predict for training or validation subsets in settings_newdata.yaml, you would have to change accordingly so:
["../coherent/in/data/processed_newdata/training/lf/", "../coherent/in/data/processed_newdata/training/hf/"]; or
["../coherent/in/data/processed_newdata/validation/lf/"] 
- before you run cnp predict, be sure to check if the files are saved to appropriate filenames

- all the plotting stuff for the mfgp is in mfgp_analzyer.py; when you run the stuff at the bottom it should automatically run it! (just be careful as coverage_summary.png is probably inaccurate)

(you can also probably use an LLM to go through the code and summarize it to an extent! sorry my code is pretty messy lol)