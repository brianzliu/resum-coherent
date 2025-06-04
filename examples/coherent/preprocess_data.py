#!/usr/bin/env python3
"""
Data preprocessing script for coherent neutrino scattering RESuM experiment.

This script converts the raw simulation data (event-level) into the format
required by RESuM, using the actual experimental design parameters.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def map_veto_to_design_params(veto_config):
    """
    Map veto configuration names to actual experimental design parameters.
    
    Based on the real experimental data:
    - Water Shielding thickness (mm)
    - Veto thickness (mm)
    """
    design_mapping = {
        # Format: veto_config -> [water_shielding_mm, veto_thickness_mm]
        'baseline': [76.2, 50],
        'Veto1': [116.2, 10],
        'Veto2': [51.2, 75],   # Note: Veto2 data not provided in simulation files
        'Veto3': [26.2, 100],
        'Veto4': [16.2, 110],
        'Veto5': [36.2, 90],
        'Veto6': [46.2, 80],
        'Veto7': [56.2, 70],
        'Veto8': [66.2, 60],
        'Veto9': [86.2, 40],
        'Veto10': [96.2, 30],
        'Veto11': [106.2, 20],
    }
    return design_mapping.get(veto_config, [76.2, 50])  # Default to baseline

def process_simulation_file(filepath, veto_config, fidelity):
    """
    Process a single simulation file to extract event-level and design-level data.
    
    Args:
        filepath: Path to the CSV file
        veto_config: Configuration name (e.g., 'baseline', 'Veto1')
        fidelity: 'lf' or 'hf'
    
    Returns:
        events_df: Event-level data for CNP training
        design_summary: Design-level metrics for MFGP
    """
    print(f"Processing {filepath}...")
    
    # Load raw simulation data
    df = pd.read_csv(filepath)
    
    # Get actual experimental design parameters for this configuration
    design_params = map_veto_to_design_params(veto_config)
    param_names = ['water_shielding_mm', 'veto_thickness_mm']
    
    # Add design parameters to each event
    for i, param_name in enumerate(param_names):
        df[param_name] = design_params[i]
    
    # Add fidelity and configuration labels
    df['fidelity'] = 1 if fidelity == 'hf' else 0
    df['config_name'] = veto_config
    
    # Event-level data for CNP (each row is one event)
    event_columns = ['water_shielding_mm', 'veto_thickness_mm',
                    'fGenX', 'fGenY', 'fGenZ', 'fEventEnergy', 'fMomentumX', 'fMomentumY', 'fMomentumZ',
                    'fEDepNR', 'fEDepVeto', 'veto_active', 'fidelity', 'config_name']
    
    events_df = df[event_columns].copy()
    
    # Design-level summary for MFGP (one row per configuration)
    total_events = len(df)
    signal_events = df['veto_active'].sum()
    design_metric = signal_events / total_events if total_events > 0 else 0.0
    
    design_summary = {
        'config_name': veto_config,
        'fidelity': 1 if fidelity == 'hf' else 0,
        'water_shielding_mm': design_params[0],
        'veto_thickness_mm': design_params[1],
        'total_events': total_events,
        'signal_events': signal_events,
        'design_metric_raw': design_metric,
        'signal_rate': design_metric,
        'background_rate': 1.0 - design_metric
    }
    
    return events_df, design_summary

def main():
    """
    Main preprocessing function.
    """
    base_dir = Path('.')
    
    # File mapping: {filename: (config_name, fidelity)}
    file_mapping = {
        # High fidelity files
        'in/data/hf/g4coherent_baseline_combined.csv': ('baseline', 'hf'),
        'in/data/hf/g4coherent_Veto1_combined.csv': ('Veto1', 'hf'),
        'in/data/hf/g4coherent_Veto3_combined.csv': ('Veto3', 'hf'),
        
        # Low fidelity files  
        'in/data/lf/g4coherent_Veto4_combined.csv': ('Veto4', 'lf'),
        'in/data/lf/g4coherent_Veto5_combined.csv': ('Veto5', 'lf'),
        'in/data/lf/g4coherent_Veto6_combined.csv': ('Veto6', 'lf'),
        'in/data/lf/g4coherent_Veto7_combined.csv': ('Veto7', 'lf'),
        'in/data/lf/g4coherent_Veto8_combined.csv': ('Veto8', 'lf'),
        'in/data/lf/g4coherent_Veto9_combined.csv': ('Veto9', 'lf'),
        'in/data/lf/g4coherent_Veto10_combined.csv': ('Veto10', 'lf'),
        'in/data/lf/g4coherent_Veto11_combined.csv': ('Veto11', 'lf'),
    }
    
    all_events = []
    all_designs = []
    
    # Process each file
    for filepath, (config_name, fidelity) in file_mapping.items():
        full_path = base_dir / filepath
        
        if full_path.exists():
            events_df, design_summary = process_simulation_file(full_path, config_name, fidelity)
            all_events.append(events_df)
            all_designs.append(design_summary)
        else:
            print(f"Warning: File not found: {full_path}")
    
    # Combine all data
    if all_events:
        combined_events = pd.concat(all_events, ignore_index=True)
        combined_designs = pd.DataFrame(all_designs)
        
        # Create output directory
        os.makedirs('processed_data', exist_ok=True)
        
        # Save processed data
        combined_events.to_csv('processed_data/event_level_data.csv', index=False)
        combined_designs.to_csv('processed_data/design_level_data.csv', index=False)
        
        print(f"\nProcessing complete!")
        print(f"Event-level data: {len(combined_events)} events across {len(all_designs)} configurations")
        print(f"Design-level data: {len(combined_designs)} design configurations")
        print(f"Files saved to processed_data/")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"High fidelity configurations: {combined_designs[combined_designs['fidelity']==1].shape[0]}")
        print(f"Low fidelity configurations: {combined_designs[combined_designs['fidelity']==0].shape[0]}")
        print(f"Average signal rate: {combined_designs['signal_rate'].mean():.4f}")
        print(f"Signal rate range: {combined_designs['signal_rate'].min():.4f} - {combined_designs['signal_rate'].max():.4f}")
        
        # Print design parameter ranges
        print(f"\nDesign Parameter Ranges:")
        print(f"Water shielding: {combined_designs['water_shielding_mm'].min():.1f} - {combined_designs['water_shielding_mm'].max():.1f} mm")
        print(f"Veto thickness: {combined_designs['veto_thickness_mm'].min():.1f} - {combined_designs['veto_thickness_mm'].max():.1f} mm")
        
    else:
        print("No files processed successfully!")

if __name__ == "__main__":
    main() 