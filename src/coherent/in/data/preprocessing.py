#!/usr/bin/env python3
"""
Script to extract resum_dataset.tar.gz and organize CSV files into appropriate directories.
File mappings are hardcoded based on the processed_newdata directory structure.
Also creates corresponding h5 files for each CSV file.
"""

import tarfile
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import h5py
except ImportError:
    print("Warning: h5py not installed. Installing now...")
    os.system("python3 -m pip install h5py --user --quiet")
    import h5py

# Define paths
BASE_DIR = Path(".")
TARBALL_PATH = BASE_DIR / "resum_dataset.tar.gz"
TEMP_EXTRACT_DIR = BASE_DIR / "temp_extracted"

# Target directories
TARGET_DIRS = {
    "training/hf": BASE_DIR / "training" / "hf",
    "training/lf": BASE_DIR / "training" / "lf",
    "validation/lf": BASE_DIR / "validation" / "lf",
}

# Hardcoded file mappings based on processed_newdata structure
TRAINING_HF_FILES = [
    "g4coherent_Veto1_combined.csv",
    "g4coherent_Veto26_combined.csv",
    "g4coherent_Veto76_combined.csv",
]

TRAINING_LF_FILES = [
    "g4coherent_Veto11_combined.csv",
    "g4coherent_Veto21_combined.csv",
    "g4coherent_Veto31_combined.csv",
    "g4coherent_Veto41_combined.csv",
    "g4coherent_Veto51_combined.csv",
    "g4coherent_Veto61_combined.csv",
    "g4coherent_Veto71_combined.csv",
    "g4coherent_Veto81_combined.csv",
    "g4coherent_Veto91_combined.csv",
]

VALIDATION_LF_FILES = [
    "g4coherent_Veto100_combined.csv",
    "g4coherent_Veto10_combined.csv",
    "g4coherent_Veto12_combined.csv",
    "g4coherent_Veto13_combined.csv",
    "g4coherent_Veto14_combined.csv",
    "g4coherent_Veto15_combined.csv",
    "g4coherent_Veto16_combined.csv",
    "g4coherent_Veto17_combined.csv",
    "g4coherent_Veto18_combined.csv",
    "g4coherent_Veto19_combined.csv",
    "g4coherent_Veto20_combined.csv",
    "g4coherent_Veto22_combined.csv",
    "g4coherent_Veto23_combined.csv",
    "g4coherent_Veto24_combined.csv",
    "g4coherent_Veto25_combined.csv",
    "g4coherent_Veto27_combined.csv",
    "g4coherent_Veto28_combined.csv",
    "g4coherent_Veto29_combined.csv",
    "g4coherent_Veto2_combined.csv",
    "g4coherent_Veto30_combined.csv",
    "g4coherent_Veto32_combined.csv",
    "g4coherent_Veto33_combined.csv",
    "g4coherent_Veto34_combined.csv",
    "g4coherent_Veto35_combined.csv",
    "g4coherent_Veto36_combined.csv",
    "g4coherent_Veto37_combined.csv",
    "g4coherent_Veto38_combined.csv",
    "g4coherent_Veto39_combined.csv",
    "g4coherent_Veto3_combined.csv",
    "g4coherent_Veto40_combined.csv",
    "g4coherent_Veto42_combined.csv",
    "g4coherent_Veto43_combined.csv",
    "g4coherent_Veto44_combined.csv",
    "g4coherent_Veto45_combined.csv",
    "g4coherent_Veto46_combined.csv",
    "g4coherent_Veto47_combined.csv",
    "g4coherent_Veto48_combined.csv",
    "g4coherent_Veto49_combined.csv",
    "g4coherent_Veto4_combined.csv",
    "g4coherent_Veto50_combined.csv",
    "g4coherent_Veto52_combined.csv",
    "g4coherent_Veto53_combined.csv",
    "g4coherent_Veto54_combined.csv",
    "g4coherent_Veto55_combined.csv",
    "g4coherent_Veto56_combined.csv",
    "g4coherent_Veto57_combined.csv",
    "g4coherent_Veto58_combined.csv",
    "g4coherent_Veto59_combined.csv",
    "g4coherent_Veto5_combined.csv",
    "g4coherent_Veto60_combined.csv",
    "g4coherent_Veto62_combined.csv",
    "g4coherent_Veto63_combined.csv",
    "g4coherent_Veto64_combined.csv",
    "g4coherent_Veto65_combined.csv",
    "g4coherent_Veto66_combined.csv",
    "g4coherent_Veto67_combined.csv",
    "g4coherent_Veto68_combined.csv",
    "g4coherent_Veto69_combined.csv",
    "g4coherent_Veto6_combined.csv",
    "g4coherent_Veto70_combined.csv",
    "g4coherent_Veto72_combined.csv",
    "g4coherent_Veto73_combined.csv",
    "g4coherent_Veto74_combined.csv",
    "g4coherent_Veto75_combined.csv",
    "g4coherent_Veto77_combined.csv",
    "g4coherent_Veto78_combined.csv",
    "g4coherent_Veto79_combined.csv",
    "g4coherent_Veto7_combined.csv",
    "g4coherent_Veto80_combined.csv",
    "g4coherent_Veto82_combined.csv",
    "g4coherent_Veto83_combined.csv",
    "g4coherent_Veto84_combined.csv",
    "g4coherent_Veto85_combined.csv",
    "g4coherent_Veto86_combined.csv",
    "g4coherent_Veto87_combined.csv",
    "g4coherent_Veto88_combined.csv",
    "g4coherent_Veto89_combined.csv",
    "g4coherent_Veto8_combined.csv",
    "g4coherent_Veto90_combined.csv",
    "g4coherent_Veto92_combined.csv",
    "g4coherent_Veto93_combined.csv",
    "g4coherent_Veto94_combined.csv",
    "g4coherent_Veto95_combined.csv",
    "g4coherent_Veto96_combined.csv",
    "g4coherent_Veto97_combined.csv",
    "g4coherent_Veto98_combined.csv",
    "g4coherent_Veto99_combined.csv",
    "g4coherent_Veto9_combined.csv",
]


def build_file_mapping():
    """Build mapping dictionary from hardcoded file lists."""
    file_mapping = {}
    
    # Map training/hf files
    for filename in TRAINING_HF_FILES:
        file_mapping[filename] = TARGET_DIRS["training/hf"]
    
    # Map training/lf files
    for filename in TRAINING_LF_FILES:
        file_mapping[filename] = TARGET_DIRS["training/lf"]
    
    # Map validation/lf files
    for filename in VALIDATION_LF_FILES:
        file_mapping[filename] = TARGET_DIRS["validation/lf"]
    
    return file_mapping


def create_h5_from_csv(csv_file_path):
    """
    Create an HDF5 file from a CSV file following the expected structure.
    
    Args:
        csv_file_path: Path to the CSV file
    
    Returns:
        Path to the created HDF5 file
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    
    # Define column categories based on settings.yaml
    theta_headers = ["water_shielding_mm", "veto_thickness_mm"]
    target_label = ["target_active"]
    weights_labels = ["weights"]
    
    # Extract theta data (first row only, since it's constant)
    theta_data = df[theta_headers].to_numpy()[0]
    
    # Extract target data
    target_data = df[target_label].to_numpy()
    
    # Extract phi data (all columns except theta, target, weights, and metadata)
    phi_headers = [col for col in df.columns if col not in 
                   theta_headers + target_label + weights_labels + 
                   ['fidelity', 'fEventID', 'fEDepNRVeto[5]', 'fEDepLAr', 
                    'source_file', 'flux_weight', 'corrected_veto_value', 
                    'veto_active', 'detector_active', 'analysis_candidate']]
    phi_data = df[phi_headers].to_numpy()
    
    # Extract weights data
    weights_data = df[weights_labels].to_numpy() if all(w in df.columns for w in weights_labels) else np.ones((len(df), 1))
    
    # Extract fidelity data (0 for LF, 1 for HF)
    # Determine fidelity based on file path
    if 'fidelity' in df.columns:
        fidelity_data = df[['fidelity']].to_numpy()
    else:
        # Infer fidelity: 1 for HF, 0 for LF
        fidelity_value = 1.0 if '/hf/' in str(csv_file_path) else 0.0
        fidelity_data = np.full((len(df), 1), fidelity_value)
    
    # Create HDF5 file path (same location as CSV)
    h5_file_path = str(csv_file_path).replace('.csv', '.h5')
    
    # Write HDF5 file
    with h5py.File(h5_file_path, "w") as hdf:
        hdf.create_dataset("fidelity", data=fidelity_data, compression="gzip")
        hdf.create_dataset("theta", data=theta_data, compression="gzip")
        hdf.create_dataset("theta_headers", data=np.array(theta_headers, dtype='S'), compression="gzip")
        hdf.create_dataset("phi", data=phi_data, compression="gzip")
        hdf.create_dataset("phi_labels", data=np.array(phi_headers, dtype='S'), compression="gzip")
        hdf.create_dataset("target", data=target_data, compression="gzip")
        hdf.create_dataset("target_labels", data=np.array(target_label, dtype='S'), compression="gzip")
        hdf.create_dataset("weights", data=weights_data, compression="gzip")
        hdf.create_dataset("weights_labels", data=np.array(weights_labels, dtype='S'), compression="gzip")
    
    return h5_file_path


def create_target_directories():
    """Create target directories if they don't exist."""
    for target_dir in TARGET_DIRS.values():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory exists: {target_dir}")


def extract_tarball():
    """Extract the tar.gz file to a temporary directory."""
    print(f"\nExtracting {TARBALL_PATH.name}...")
    
    # Remove temp directory if it exists
    if TEMP_EXTRACT_DIR.exists():
        shutil.rmtree(TEMP_EXTRACT_DIR)
    
    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract the tar.gz file
    with tarfile.open(TARBALL_PATH, 'r:gz') as tar:
        tar.extractall(TEMP_EXTRACT_DIR)
    
    print(f"✓ Extracted to {TEMP_EXTRACT_DIR}")


def move_csv_files(file_mapping):
    """Move CSV files from extracted archive to target directories and create h5 files."""
    moved_count = {"training/hf": 0, "training/lf": 0, "validation/lf": 0}
    h5_count = {"training/hf": 0, "training/lf": 0, "validation/lf": 0}
    not_found_count = 0
    h5_errors = []
    
    # Walk through the extracted directory
    for root, dirs, files in os.walk(TEMP_EXTRACT_DIR):
        for filename in files:
            # Only process CSV files
            if not filename.endswith('.csv'):
                continue
            
            source_path = Path(root) / filename
            
            # Look up the target directory in our mapping
            if filename in file_mapping:
                target_dir = file_mapping[filename]
                destination = target_dir / filename
                shutil.copy2(source_path, destination)
                
                # Track which category this file belongs to
                category = None
                if target_dir == TARGET_DIRS["training/hf"]:
                    moved_count["training/hf"] += 1
                    category = "training/hf"
                elif target_dir == TARGET_DIRS["training/lf"]:
                    moved_count["training/lf"] += 1
                    category = "training/lf"
                elif target_dir == TARGET_DIRS["validation/lf"]:
                    moved_count["validation/lf"] += 1
                    category = "validation/lf"
                    
                print(f"  ✓ CSV: {filename} → {target_dir.relative_to(BASE_DIR)}/")
                
                # Create corresponding h5 file
                try:
                    h5_path = create_h5_from_csv(destination)
                    h5_count[category] += 1
                    h5_filename = Path(h5_path).name
                    print(f"  ✓ H5:  {h5_filename} created")
                except Exception as e:
                    error_msg = f"{filename}: {str(e)}"
                    h5_errors.append(error_msg)
                    print(f"  ✗ H5:  Failed to create h5 file - {e}")
            else:
                print(f"  ⚠ {filename} - not in mapping (skipped)")
                not_found_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  training/hf:    {moved_count['training/hf']:3d} CSV, {h5_count['training/hf']:3d} H5 files")
    print(f"  training/lf:    {moved_count['training/lf']:3d} CSV, {h5_count['training/lf']:3d} H5 files")
    print(f"  validation/lf:  {moved_count['validation/lf']:3d} CSV, {h5_count['validation/lf']:3d} H5 files")
    print(f"  {'─'*60}")
    print(f"  Total:          {sum(moved_count.values()):3d} CSV, {sum(h5_count.values()):3d} H5 files")
    if not_found_count > 0:
        print(f"  Skipped:        {not_found_count:3d} files (not in mapping)")
    if h5_errors:
        print(f"  H5 Errors:      {len(h5_errors):3d} files failed")
        print(f"\nH5 creation errors:")
        for error in h5_errors[:5]:  # Show first 5 errors
            print(f"    - {error}")
        if len(h5_errors) > 5:
            print(f"    ... and {len(h5_errors) - 5} more")
    print(f"{'='*60}")


def cleanup():
    """Remove the temporary extraction directory."""
    if TEMP_EXTRACT_DIR.exists():
        shutil.rmtree(TEMP_EXTRACT_DIR)
        print(f"\n✓ Cleaned up temporary directory")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Dataset Extraction and Organization Script")
    print("=" * 60)
    print(f"Files will be organized into:")
    print(f"  - training/hf:   {len(TRAINING_HF_FILES)} CSV + H5 files")
    print(f"  - training/lf:   {len(TRAINING_LF_FILES)} CSV + H5 files")
    print(f"  - validation/lf: {len(VALIDATION_LF_FILES)} CSV + H5 files")
    print(f"  - Total:         {len(TRAINING_HF_FILES) + len(TRAINING_LF_FILES) + len(VALIDATION_LF_FILES)} files × 2 formats")
    print("=" * 60)
    
    # Check if tarball exists
    if not TARBALL_PATH.exists():
        print(f"\n✗ Error: Tarball not found at {TARBALL_PATH}")
        return
    
    try:
        # Step 1: Build file mapping
        print("\n[Step 1/4] Building file mapping...")
        file_mapping = build_file_mapping()
        print(f"✓ Mapped {len(file_mapping)} files")
        
        # Step 2: Create target directories
        print("\n[Step 2/4] Creating target directories...")
        create_target_directories()
        
        # Step 3: Extract tarball
        print("\n[Step 3/4] Extracting tarball...")
        extract_tarball()
        
        # Step 4: Move CSV files and create H5 files
        print("\n[Step 4/4] Moving CSV files and creating H5 files...")
        move_csv_files(file_mapping)
        
        # Step 5: Cleanup
        cleanup()
        
        print("\n" + "=" * 60)
        print("✓ Complete! Files have been organized successfully.")
        print("✓ CSV and H5 files are ready for use.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        # Try to cleanup even if there's an error
        if TEMP_EXTRACT_DIR.exists():
            shutil.rmtree(TEMP_EXTRACT_DIR)


if __name__ == "__main__":
    main()
