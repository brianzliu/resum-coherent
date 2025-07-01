#!/usr/bin/env python3
"""
Enhanced program to compare all datasets in HDF5 files versus their corresponding CSV files.
This includes both original and mixedup datasets.
"""

import os
import pandas as pd
import h5py
import glob
from pathlib import Path

def count_csv_rows(file_path):
    """Count the number of rows in a CSV file (excluding header)."""
    try:
        df = pd.read_csv(file_path)
        return len(df)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return -1

def analyze_hdf5_datasets(file_path):
    """Analyze all relevant datasets in an HDF5 file."""
    results = {}
    try:
        with h5py.File(file_path, 'r') as f:
            # Check for main datasets
            datasets_to_check = ['target', 'phi', 'fidelity', 'target_mixedup', 'phi_mixedup']
            
            for dataset_name in datasets_to_check:
                if dataset_name in f:
                    dataset = f[dataset_name]
                    if hasattr(dataset, 'shape') and len(dataset.shape) > 0:
                        results[dataset_name] = dataset.shape[0]
                    else:
                        results[dataset_name] = 0
                else:
                    results[dataset_name] = None  # Dataset doesn't exist
            
            # Also check what the first key is (what our original comparison used)
            keys = list(f.keys())
            if keys:
                first_key = keys[0]
                if first_key not in datasets_to_check:
                    first_dataset = f[first_key]
                    if hasattr(first_dataset, 'shape') and len(first_dataset.shape) > 0:
                        results[f'first_key({first_key})'] = first_dataset.shape[0]
                        
    except Exception as e:
        print(f"Error reading HDF5 file {file_path}: {e}")
        return {}
    
    return results

def find_file_pairs(processed_dir):
    """Find all HDF5 files and their corresponding CSV files."""
    pairs = []
    
    # Find all HDF5 files recursively
    hdf5_pattern = os.path.join(processed_dir, '**', '*.h5')
    hdf5_files = glob.glob(hdf5_pattern, recursive=True)
    
    for hdf5_file in hdf5_files:
        # Generate corresponding CSV file path
        csv_file = hdf5_file.replace('.h5', '.csv')
        
        if os.path.exists(csv_file):
            pairs.append((hdf5_file, csv_file))
        else:
            print(f"Warning: No corresponding CSV file found for {hdf5_file}")
    
    return pairs

def main():
    """Main function to compare all datasets in HDF5 files with CSV files."""
    processed_dir = "/home/bliu4/resum-coherent2/examples/coherent/in/data/processed"
    
    if not os.path.exists(processed_dir):
        print(f"Error: Directory {processed_dir} does not exist!")
        return
    
    print("Analyzing all datasets in HDF5 files vs CSV files...")
    print("=" * 100)
    
    file_pairs = find_file_pairs(processed_dir)
    
    if not file_pairs:
        print("No HDF5-CSV file pairs found!")
        return
    
    for hdf5_file, csv_file in file_pairs:
        rel_hdf5 = os.path.relpath(hdf5_file, processed_dir)
        rel_csv = os.path.relpath(csv_file, processed_dir)
        
        print(f"\nFile: {rel_hdf5.replace('.h5', '')}")
        print("-" * 60)
        
        # Get CSV row count
        csv_count = count_csv_rows(csv_file)
        print(f"CSV rows: {csv_count:,}")
        
        # Get HDF5 dataset counts
        hdf5_datasets = analyze_hdf5_datasets(hdf5_file)
        
        print("HDF5 datasets:")
        for dataset_name, count in hdf5_datasets.items():
            if count is not None:
                if count == csv_count:
                    match_symbol = "✓"
                else:
                    match_symbol = "✗"
                print(f"  {dataset_name:20}: {count:,} {match_symbol}")
            else:
                print(f"  {dataset_name:20}: NOT FOUND")
        
        # Analysis
        print("\nAnalysis:")
        if 'target' in hdf5_datasets and hdf5_datasets['target'] == csv_count:
            print("  ✓ Original target dataset matches CSV")
        elif 'target' in hdf5_datasets and hdf5_datasets['target'] is not None:
            print(f"  ✗ Original target dataset ({hdf5_datasets['target']:,}) differs from CSV ({csv_count:,})")
        
        if 'target_mixedup' in hdf5_datasets and hdf5_datasets['target_mixedup'] is not None:
            print(f"  → Mixedup dataset has {hdf5_datasets['target_mixedup']:,} rows (reduced from original)")
            
        print("=" * 100)

if __name__ == "__main__":
    main() 