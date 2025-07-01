#!/usr/bin/env python3
"""
Sanity Check Script for Data Files
Checks CSV and H5 files for data integrity issues that could cause empty batches
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import h5py, but make it optional
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not available. H5 file checking will be skipped.")

def check_csv_files(data_dir):
    """Check all CSV files for data integrity issues"""
    print("="*60)
    print("CHECKING CSV FILES")
    print("="*60)
    
    csv_files = list(Path(data_dir).glob("*.csv"))
    if not csv_files:
        print("No CSV files found in directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to check...")
    
    issues_found = []
    file_stats = {}
    
    for i, csv_file in enumerate(csv_files):
        print(f"\n[{i+1}/{len(csv_files)}] Checking: {csv_file.name}")
        
        try:
            # Try to read the CSV
            df = pd.read_csv(csv_file)
            
            # Basic stats
            file_stats[csv_file.name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'has_nan': df.isnull().any().any(),
                'nan_count': df.isnull().sum().sum(),
                'empty_rows': (df.isnull().all(axis=1)).sum(),
                'duplicate_rows': df.duplicated().sum()
            }
            
            # Check for issues
            issues = []
            
            # Empty file
            if df.empty:
                issues.append("FILE IS EMPTY")
            
            # Very few rows
            elif len(df) < 10:
                issues.append(f"VERY FEW ROWS: Only {len(df)} rows")
            
            # Missing columns
            if df.shape[1] == 0:
                issues.append("NO COLUMNS FOUND")
            
            # Check for NaN values
            if df.isnull().any().any():
                nan_cols = df.columns[df.isnull().any()].tolist()
                issues.append(f"NaN VALUES in columns: {nan_cols}")
            
            # Check for completely empty rows
            empty_rows = (df.isnull().all(axis=1)).sum()
            if empty_rows > 0:
                issues.append(f"EMPTY ROWS: {empty_rows} completely empty rows")
            
            # Check for infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                inf_check = np.isinf(df[numeric_cols]).any().any()
                if inf_check:
                    inf_cols = numeric_cols[np.isinf(df[numeric_cols]).any()].tolist()
                    issues.append(f"INFINITE VALUES in columns: {inf_cols}")
            
            # Check for columns with all same values
            constant_cols = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)
            if constant_cols:
                issues.append(f"CONSTANT COLUMNS: {constant_cols}")
            
            if issues:
                issues_found.append({
                    'file': csv_file.name,
                    'issues': issues
                })
            
            # Print summary
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   NaN count: {df.isnull().sum().sum()}")
            if issues:
                print(f"   ⚠️  ISSUES: {'; '.join(issues)}")
            else:
                print(f"   ✅ No issues found")
                
        except Exception as e:
            error_msg = f"ERROR READING FILE: {str(e)}"
            issues_found.append({
                'file': csv_file.name,
                'issues': [error_msg]
            })
            print(f"   ❌ {error_msg}")
    
    return issues_found, file_stats

def check_h5_files(data_dir):
    """Check all H5 files for data integrity issues"""
    print("\n" + "="*60)
    print("CHECKING H5 FILES")
    print("="*60)
    
    if not HAS_H5PY:
        print("❌ h5py not available. Cannot check H5 files.")
        print("Install h5py with: pip install h5py")
        return [], {}
    
    h5_files = list(Path(data_dir).glob("*.h5"))
    if not h5_files:
        print("No H5 files found in directory.")
        return [], {}
    
    print(f"Found {len(h5_files)} H5 files to check...")
    
    issues_found = []
    file_stats = {}
    
    for i, h5_file in enumerate(h5_files):
        print(f"\n[{i+1}/{len(h5_files)}] Checking: {h5_file.name}")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                datasets = list(f.keys())
                
                file_stats[h5_file.name] = {
                    'datasets': datasets,
                    'file_size': h5_file.stat().st_size
                }
                
                issues = []
                
                # Check if file has expected datasets
                expected_datasets = ['phi', 'theta', 'target']
                missing_datasets = [ds for ds in expected_datasets if ds not in datasets]
                if missing_datasets:
                    issues.append(f"MISSING DATASETS: {missing_datasets}")
                
                # Check each dataset
                for dataset_name in datasets:
                    try:
                        data = f[dataset_name]
                        shape = data.shape
                        dtype = data.dtype
                        
                        print(f"   Dataset '{dataset_name}': shape={shape}, dtype={dtype}")
                        
                        # Check for empty datasets
                        if data.size == 0:
                            issues.append(f"EMPTY DATASET: {dataset_name}")
                            continue
                        
                        # Check for very small datasets
                        if data.shape[0] < 10:
                            issues.append(f"VERY SMALL DATASET: {dataset_name} has only {data.shape[0]} rows")
                        
                        # Sample some data to check for NaN/inf
                        if data.size > 0:
                            sample_size = min(1000, data.size)
                            if len(shape) > 1:
                                sample = data[:min(100, shape[0]), :]
                            else:
                                sample = data[:min(1000, shape[0])]
                            
                            sample_array = np.array(sample)
                            
                            # Check for NaN
                            if np.isnan(sample_array).any():
                                issues.append(f"NaN VALUES in dataset: {dataset_name}")
                            
                            # Check for inf
                            if np.isinf(sample_array).any():
                                issues.append(f"INFINITE VALUES in dataset: {dataset_name}")
                    
                    except Exception as e:
                        issues.append(f"ERROR READING DATASET {dataset_name}: {str(e)}")
                
                # Store stats
                file_stats[h5_file.name].update({
                    'issues_count': len(issues)
                })
                
                if issues:
                    issues_found.append({
                        'file': h5_file.name,
                        'issues': issues
                    })
                
                if issues:
                    print(f"   ⚠️  ISSUES: {'; '.join(issues)}")
                else:
                    print(f"   ✅ No issues found")
                
        except Exception as e:
            error_msg = f"ERROR READING FILE: {str(e)}"
            issues_found.append({
                'file': h5_file.name,
                'issues': [error_msg]
            })
            print(f"   ❌ {error_msg}")
    
    return issues_found, file_stats

def check_data_consistency(data_dir):
    """Check for consistency across files"""
    print("\n" + "="*60)
    print("CHECKING DATA CONSISTENCY ACROSS FILES")
    print("="*60)
    
    issues = []
    
    # Check H5 files for consistent shapes
    h5_files = list(Path(data_dir).glob("*.h5"))
    if h5_files and HAS_H5PY:
        shapes = {}
        for h5_file in h5_files[:5]:  # Check first 5 files for shapes
            try:
                with h5py.File(h5_file, 'r') as f:
                    for dataset in ['phi', 'theta', 'target']:
                        if dataset in f:
                            shape = f[dataset].shape
                            if dataset not in shapes:
                                shapes[dataset] = []
                            shapes[dataset].append((h5_file.name, shape))
            except:
                continue
        
        # Check for inconsistent shapes
        for dataset, file_shapes in shapes.items():
            if len(file_shapes) > 1:
                unique_shapes = set([shape for _, shape in file_shapes])
                if len(unique_shapes) > 1:
                    print(f"⚠️  INCONSISTENT SHAPES for dataset '{dataset}':")
                    for file_name, shape in file_shapes:
                        print(f"   {file_name}: {shape}")
                    issues.append(f"Inconsistent shapes in dataset {dataset}")
    
    return issues

def diagnose_empty_batch_causes(data_dir):
    """Specific diagnosis for empty batch causes"""
    print("\n" + "="*60)
    print("DIAGNOSING EMPTY BATCH CAUSES")
    print("="*60)
    
    if not HAS_H5PY:
        print("❌ h5py not available. Cannot diagnose H5 loading issues.")
        return
    
    h5_files = list(Path(data_dir).glob("*.h5"))
    if not h5_files:
        print("No H5 files found.")
        return
    
    print(f"Analyzing {len(h5_files)} H5 files for empty batch causes...")
    
    # Simulate the HDF5Dataset loading process
    batch_size = 3000
    files_per_batch = 20
    
    for i, h5_file in enumerate(h5_files):
        print(f"\n[{i+1}/{len(h5_files)}] Analyzing: {h5_file.name}")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                # Get main datasets
                phi = f['phi'] if 'phi' in f else None
                theta = f['theta'] if 'theta' in f else None
                target = f['target'] if 'target' in f else None
                
                if phi is None or target is None:
                    print("   ❌ CRITICAL: Missing phi or target datasets")
                    continue
                
                print(f"   📊 phi shape: {phi.shape}")
                print(f"   📊 target shape: {target.shape}")
                if theta is not None:
                    print(f"   📊 theta shape: {theta.shape}")
                
                # Simulate rows_per_file calculation
                rows_per_file = batch_size // files_per_batch
                print(f"   📊 rows_per_file: {rows_per_file}")
                
                # Calculate how many epochs this file can support
                max_epochs = phi.shape[0] // rows_per_file
                print(f"   📊 max_epochs supported: {max_epochs}")
                
                # Check various epoch scenarios
                problematic_epochs = []
                for epoch in range(min(10, max_epochs + 5)):  # Check first 10 epochs or beyond max
                    start_idx = epoch * rows_per_file
                    end_idx = start_idx + rows_per_file
                    
                    if start_idx >= phi.shape[0]:
                        problematic_epochs.append(f"epoch {epoch}: start_idx {start_idx} >= file_length {phi.shape[0]}")
                    elif end_idx > phi.shape[0]:
                        actual_rows = phi.shape[0] - start_idx
                        if actual_rows <= 0:
                            problematic_epochs.append(f"epoch {epoch}: no valid rows (start_idx={start_idx}, file_length={phi.shape[0]})")
                        else:
                            print(f"   ⚠️  epoch {epoch}: partial data ({actual_rows} rows instead of {rows_per_file})")
                
                if problematic_epochs:
                    print("   ❌ EMPTY BATCH SCENARIOS:")
                    for prob in problematic_epochs:
                        print(f"      {prob}")
                else:
                    print("   ✅ No obvious empty batch scenarios in first 10 epochs")
                
                # Check for specific data issues
                phi_sample = phi[:min(1000, phi.shape[0]), :]
                target_sample = target[:min(1000, target.shape[0])]
                
                # Check for any obviously bad data
                phi_array = np.array(phi_sample)
                target_array = np.array(target_sample)
                
                phi_issues = []
                if np.any(np.isnan(phi_array)):
                    phi_issues.append("NaN values")
                if np.any(np.isinf(phi_array)):
                    phi_issues.append("Infinite values")
                if np.all(phi_array == 0):
                    phi_issues.append("All zeros")
                
                target_issues = []
                if target_array.ndim > 1:
                    target_flat = target_array.flatten()
                else:
                    target_flat = target_array
                if np.any(np.isnan(target_flat)):
                    target_issues.append("NaN values")
                if np.any(np.isinf(target_flat)):
                    target_issues.append("Infinite values")
                
                if phi_issues:
                    print(f"   ⚠️  phi data issues: {', '.join(phi_issues)}")
                if target_issues:
                    print(f"   ⚠️  target data issues: {', '.join(target_issues)}")
                
                if not phi_issues and not target_issues and not problematic_epochs:
                    print("   ✅ File appears healthy for batch loading")
                    
        except Exception as e:
            print(f"   ❌ ERROR analyzing file: {str(e)}")

def generate_summary_report(csv_issues, csv_stats, h5_issues, h5_stats, consistency_issues, data_dir):
    """Generate a summary report"""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    total_csv_files = len(csv_stats) if csv_stats else 0
    total_h5_files = len(h5_stats) if h5_stats else 0
    total_csv_issues = len(csv_issues) if csv_issues else 0
    total_h5_issues = len(h5_issues) if h5_issues else 0
    
    print(f"Directory checked: {data_dir}")
    print(f"Total CSV files: {total_csv_files}")
    print(f"Total H5 files: {total_h5_files}")
    print(f"CSV files with issues: {total_csv_issues}")
    print(f"H5 files with issues: {total_h5_issues}")
    print(f"Consistency issues: {len(consistency_issues)}")
    
    print("\n" + "-"*60)
    print("FILES WITH CRITICAL ISSUES:")
    print("-"*60)
    
    critical_files = []
    
    # CSV issues
    if csv_issues:
        for file_issue in csv_issues:
            critical_issues = [issue for issue in file_issue['issues'] 
                             if any(keyword in issue for keyword in ['EMPTY', 'ERROR', 'NO COLUMNS'])]
            if critical_issues:
                critical_files.append(f"CSV: {file_issue['file']} - {'; '.join(critical_issues)}")
    
    # H5 issues
    if h5_issues:
        for file_issue in h5_issues:
            critical_issues = [issue for issue in file_issue['issues'] 
                             if any(keyword in issue for keyword in ['EMPTY', 'ERROR', 'MISSING'])]
            if critical_issues:
                critical_files.append(f"H5: {file_issue['file']} - {'; '.join(critical_issues)}")
    
    if critical_files:
        for critical_file in critical_files:
            print(f"❌ {critical_file}")
    else:
        print("✅ No critical issues found!")
    
    # Recommendations
    print("\n" + "-"*60)
    print("RECOMMENDATIONS:")
    print("-"*60)
    
    if critical_files:
        print("1. Remove or fix files with critical issues")
        print("2. Check your data pipeline for files that are not being generated properly")
        print("3. Consider adding validation in your data generation process")
    
    if total_csv_issues > 0 or total_h5_issues > 0:
        print("4. Review files with NaN or infinite values")
        print("5. Consider data cleaning/preprocessing steps")
    
    if len(consistency_issues) > 0:
        print("6. Ensure all files have consistent data formats and shapes")
    
    print("7. Consider adding try-catch blocks in your data loading code")
    print("8. Implement empty batch detection in your training loop")

def main():
    # Directory to check
    data_dir = "/home/bliu4/resum-coherent2/examples/coherent/in/data/processed/hf"
    
    print("DATA INTEGRITY SANITY CHECK")
    print("="*60)
    print(f"Checking directory: {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Directory {data_dir} does not exist!")
        return
    
    # Check CSV files
    csv_issues, csv_stats = check_csv_files(data_dir)
    
    # Check H5 files
    h5_issues, h5_stats = check_h5_files(data_dir)
    
    # Check consistency
    consistency_issues = check_data_consistency(data_dir)
    
    # NEW: Diagnose empty batch causes
    diagnose_empty_batch_causes(data_dir)
    
    # Generate summary
    generate_summary_report(csv_issues, csv_stats, h5_issues, h5_stats, consistency_issues, data_dir)

if __name__ == "__main__":
    main()
