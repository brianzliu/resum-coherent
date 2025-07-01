#!/usr/bin/env python3
"""
Script to inspect the internal structure of HDF5 files to understand 
why different programs are getting different row counts.
"""

import h5py
import os
import glob

def inspect_hdf5_file(file_path):
    """Inspect the structure and datasets in an HDF5 file."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("Available datasets and groups:")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset: {name}")
                    print(f"    Shape: {obj.shape}")
                    print(f"    Dtype: {obj.dtype}")
                    print(f"    Size: {obj.size}")
                    if len(obj.shape) > 0:
                        print(f"    Number of rows: {obj.shape[0]}")
                elif isinstance(obj, h5py.Group):
                    print(f"  Group: {name}")
            
            f.visititems(print_structure)
            
            # Check specifically for common dataset names
            common_names = ['data', 'dataset', 'table', 'df', 'target', 'phi', 'theta']
            print(f"\nChecking for common dataset names:")
            for name in common_names:
                if name in f:
                    dataset = f[name]
                    print(f"  {name}: shape={dataset.shape}, rows={dataset.shape[0] if len(dataset.shape) > 0 else 'N/A'}")
                else:
                    print(f"  {name}: NOT FOUND")
            
            # Get the first key (what our comparison script would use)
            keys = list(f.keys())
            if keys:
                first_key = keys[0]
                first_dataset = f[first_key]
                print(f"\nFirst key analysis:")
                print(f"  First key: {first_key}")
                print(f"  Shape: {first_dataset.shape}")
                print(f"  Rows: {first_dataset.shape[0] if len(first_dataset.shape) > 0 else 'N/A'}")
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def main():
    """Main function to inspect HDF5 files."""
    processed_dir = "/home/bliu4/resum-coherent2/examples/coherent/in/data/processed"
    
    # Focus on the problematic files from the lf directory
    problematic_files = [
        "lf/g4coherent_Veto5_combined_with_theta.h5",
        "lf/g4coherent_Veto6_combined_with_theta.h5", 
        "lf/g4coherent_Veto9_combined_with_theta.h5",
        "lf/g4coherent_Veto10_combined_with_theta.h5"
    ]
    
    print("Inspecting HDF5 file structures to understand row count discrepancies...")
    
    for rel_file in problematic_files:
        full_path = os.path.join(processed_dir, rel_file)
        if os.path.exists(full_path):
            inspect_hdf5_file(full_path)
        else:
            print(f"File not found: {full_path}")
    
    # Also check one file from hf directory for comparison
    hf_file = os.path.join(processed_dir, "hf/g4coherent_Veto1_combined_with_theta.h5")
    if os.path.exists(hf_file):
        inspect_hdf5_file(hf_file)

if __name__ == "__main__":
    main() 