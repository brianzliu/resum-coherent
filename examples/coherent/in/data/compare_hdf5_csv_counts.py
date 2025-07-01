#!/usr/bin/env python3
"""
Program to compare the number of data points/lines in HDF5 files 
versus their corresponding CSV files in the processed directory.
"""

import os
import pandas as pd
import h5py
import glob
from pathlib import Path

def count_hdf5_rows(file_path):
    """Count the number of rows in an HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            # Try to find the main dataset - common names are 'data', 'dataset', or the first key
            keys = list(f.keys())
            if not keys:
                return 0
            
            # Try common dataset names first
            main_key = None
            for common_name in ['data', 'dataset', 'table', 'df']:
                if common_name in keys:
                    main_key = common_name
                    break
            
            # If no common name found, use the first key
            if main_key is None:
                main_key = keys[0]
            
            dataset = f[main_key]
            
            # Handle different dataset types
            if hasattr(dataset, 'shape'):
                return dataset.shape[0]
            elif hasattr(dataset, '__len__'):
                return len(dataset)
            else:
                return 0
                
    except Exception as e:
        print(f"Error reading HDF5 file {file_path}: {e}")
        return -1

def count_csv_rows(file_path):
    """Count the number of rows in a CSV file (excluding header)."""
    try:
        df = pd.read_csv(file_path)
        return len(df)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return -1

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
    """Main function to compare HDF5 and CSV file data counts."""
    processed_dir = "/home/bliu4/resum-coherent2/examples/coherent/in/data/processed"
    
    if not os.path.exists(processed_dir):
        print(f"Error: Directory {processed_dir} does not exist!")
        return
    
    print("Comparing HDF5 and CSV file data counts...")
    print("=" * 80)
    
    file_pairs = find_file_pairs(processed_dir)
    
    if not file_pairs:
        print("No HDF5-CSV file pairs found!")
        return
    
    results = []
    
    for hdf5_file, csv_file in file_pairs:
        rel_hdf5 = os.path.relpath(hdf5_file, processed_dir)
        rel_csv = os.path.relpath(csv_file, processed_dir)
        
        hdf5_count = count_hdf5_rows(hdf5_file)
        csv_count = count_csv_rows(csv_file)
        
        if hdf5_count == -1 or csv_count == -1:
            continue
        
        difference = abs(hdf5_count - csv_count)
        match = "✓" if hdf5_count == csv_count else "✗"
        
        results.append({
            'file_name': rel_hdf5.replace('.h5', ''),
            'hdf5_count': hdf5_count,
            'csv_count': csv_count,
            'difference': difference,
            'match': match
        })
        
        print(f"File: {rel_hdf5.replace('.h5', '')}")
        print(f"  HDF5 rows: {hdf5_count:,}")
        print(f"  CSV rows:  {csv_count:,}")
        print(f"  Difference: {difference:,}")
        print(f"  Match: {match}")
        print("-" * 40)
    
    # Summary
    print("\nSUMMARY:")
    print("=" * 80)
    
    if results:
        total_files = len(results)
        matching_files = sum(1 for r in results if r['difference'] == 0)
        mismatched_files = total_files - matching_files
        
        print(f"Total file pairs analyzed: {total_files}")
        print(f"Files with matching counts: {matching_files}")
        print(f"Files with mismatched counts: {mismatched_files}")
        
        if mismatched_files > 0:
            print(f"\nFiles with mismatches:")
            for result in results:
                if result['difference'] > 0:
                    print(f"  {result['file_name']}: HDF5({result['hdf5_count']:,}) vs CSV({result['csv_count']:,}) - diff: {result['difference']:,}")
        
        total_hdf5_rows = sum(r['hdf5_count'] for r in results)
        total_csv_rows = sum(r['csv_count'] for r in results)
        print(f"\nTotal HDF5 rows across all files: {total_hdf5_rows:,}")
        print(f"Total CSV rows across all files: {total_csv_rows:,}")
        print(f"Overall difference: {abs(total_hdf5_rows - total_csv_rows):,}")
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main() 