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
- `veto_active`: Binary outcome (1 = signal detected, 0 = background)

## Experimental Configurations

The actual design parameters for each configuration:

| Configuration | Water Shielding (mm) | Veto Thickness (mm) | Fidelity |
|--------------|---------------------|-------------------|----------|
| baseline     | 76.2               | 50                | HF       |
| Veto1        | 116.2              | 10                | HF       |
| Veto3        | 26.2               | 100               | HF       |
| Veto4        | 16.2               | 110               | LF       |
| Veto5        | 36.2               | 90                | LF       |
| Veto6        | 46.2               | 80                | LF       |
| Veto7        | 56.2               | 70                | LF       |
| Veto8        | 66.2               | 60                | LF       |
| Veto9        | 86.2               | 40                | LF       |
| Veto10       | 96.2               | 30                | LF       |
| Veto11       | 106.2              | 20                | LF       |

## Data Structure

### Current Data Files
The simulation files (`g4coherent_*.csv`) contain event-level physics data for different veto configurations. Each row represents one simulated particle event with its physics parameters and outcome.

### Fidelity Levels
- **Low Fidelity (LF)**: Smaller datasets with simplified physics (Veto4-Veto11)
- **High Fidelity (HF)**: Larger datasets with full physics simulation (baseline, Veto1, Veto3)

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
│   │   ├── lf/          # Low fidelity simulations (8 files)
│   │   └── hf/          # High fidelity simulations (3 files)
│   └── mfgp/            # Multi-fidelity GP inputs
├── out/
│   ├── cnp/             # CNP model outputs
│   ├── mfgp/            # MFGP optimization results
│   └── pce/             # Polynomial Chaos Expansion outputs
├── settings.yaml        # Configuration parameters
├── inequalities.py      # Design constraints
└── README.md           # This file
```

## Key Differences from LEGEND Neutron Moderator

- **Application Domain**: Coherent neutrino scattering vs neutron background reduction
- **Design Parameters**: Water shielding + veto thickness vs moderator geometry (radius, thickness, panels, angle, length)
- **Physics**: Neutrino-nucleus interactions vs neutron moderation
- **Constraints**: Shielding effectiveness vs geometric feasibility

## Next Steps

1. **Data Preprocessing**: Run `preprocess_data.py` to map configurations to design parameters
2. **CNP Implementation**: Adapt training for coherent scattering physics
3. **Constraint Refinement**: Adjust inequalities for shielding design constraints
4. **Validation**: Test with coherent scattering simulation data

## References

- RESuM Paper: "Rare Event Surrogate Model for Physics Detector Design" (ICLR 2025)
- LEGEND Collaboration neutron moderator optimization example
- Coherent neutrino scattering simulation framework 