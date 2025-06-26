import pandas as pd
import os
import glob
import re
import argparse
from tqdm import tqdm
import numpy as np
import h5py

# Mapping of veto configurations to theta parameters
veto_config_mapping = {
    'baseline': [76.2, 50],
    'Veto1': [116.2, 10],
    'Veto2': [51.2, 75],
    'Veto3': [26.2, 100],
    'Veto4': [16.2, 110],
    'Veto5': [36.2, 90],
    'Veto6': [46.2, 80],
    'Veto7': [56.2, 70],
    'Veto8': [66.2, 60],
    'Veto9': [86.2, 40],
    'Veto10': [96.2, 30],
    'Veto11': [106.2, 20]
}

def extract_veto_config_from_filename(filename):
    """Extract veto configuration from filename using regex for accurate matching"""
    basename = os.path.basename(filename).lower()
    
    # Use regex to match baseline first
    if re.search(r'\bbaseline\b', basename):
        return 'baseline'
    
    # Use regex to match Veto configurations with word boundaries
    # This will match Veto followed by digits, ensuring exact matches
    veto_match = re.search(r'\bveto(\d+)\b', basename)
    if veto_match:
        veto_num = veto_match.group(1)
        config_key = f'Veto{veto_num}'
        if config_key in veto_config_mapping:
            return config_key
    
    # Fallback: try direct string matching for any missed cases
    for config in sorted(veto_config_mapping.keys(), key=len, reverse=True):
        if config.lower() in basename:
            return config
    
    return None

def create_output_directories(base_data_dir):
    """Create organized output directories for processed files"""
    processed_dir = os.path.join(base_data_dir, 'processed')
    lf_dir = os.path.join(processed_dir, 'lf')
    hf_dir = os.path.join(processed_dir, 'hf')
    
    os.makedirs(lf_dir, exist_ok=True)
    os.makedirs(hf_dir, exist_ok=True)
    
    return processed_dir, lf_dir, hf_dir

def determine_output_path(input_file_path, base_data_dir):
    """Determine the appropriate output path based on input file location"""
    processed_dir, lf_dir, hf_dir = create_output_directories(base_data_dir)
    
    # Determine if file is from lf or hf directory
    if '/lf/' in input_file_path:
        output_dir = lf_dir
    elif '/hf/' in input_file_path:
        output_dir = hf_dir
    else:
        output_dir = processed_dir
    
    # Create new filename
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_filename = f"{base_name}_with_theta.csv"
    
    return os.path.join(output_dir, output_filename)

def add_theta_headers_to_csv(input_file_path, base_data_dir, output_file_path=None):
    """Add theta headers to CSV file based on veto configuration"""
    
    # Extract configuration from filename
    config = extract_veto_config_from_filename(input_file_path)
    if config is None:
        print(f"Warning: Could not determine veto configuration for {input_file_path}")
        return None
    
    # Get theta values for this configuration
    water_shielding_mm, veto_thickness_mm = veto_config_mapping[config]
    
    # Read the original CSV
    print(f"Reading {input_file_path}...")
    df = pd.read_csv(input_file_path)
    
    # Add theta columns after the first column (index 1 and 2)
    df.insert(1, 'water_shielding_mm', water_shielding_mm)
    df.insert(2, 'veto_thickness_mm', veto_thickness_mm)
    
    # Determine output path
    if output_file_path is None:
        output_file_path = determine_output_path(input_file_path, base_data_dir)
    
    # Save the modified CSV
    df.to_csv(output_file_path, index=False)
    print(f"Processed {config}: {os.path.basename(input_file_path)} -> {output_file_path}")
    
    return df

def add_weights_column_to_csv(csv_file_path):
    """
    Add a 'weights' column with value 1 to a CSV file and save it in place.
    
    Args:
        csv_file_path (str): Path to the CSV file to modify
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if 'weights' column already exists
        if 'weights' in df.columns:
            print(f"  'weights' column already exists in {os.path.basename(csv_file_path)}")
            return
        
        # Add the 'weights' column with value 1
        df['weights'] = 1
        
        # Save the modified dataframe back to the same file
        df.to_csv(csv_file_path, index=False)
        print(f"  Added 'weights' column to {os.path.basename(csv_file_path)}")
        
    except Exception as e:
        print(f"  Error processing {csv_file_path}: {e}")

def convert_single_csv_to_hdf5(csv_file, hdf5_file, theta_headers, target_label, weights_labels):
    """
    Convert a single CSV file to HDF5 format.
    Adapted from utilities.py for the COHERENT data structure.
    
    Args:
        csv_file (str): Path to input CSV file
        hdf5_file (str): Path to output HDF5 file
        theta_headers (list): Column names for theta parameters
        target_label (list): Column names for target variables
        weights_labels (list): Column names for weights
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # **Extract 'theta' data (design parameters)**
        theta_data = df[theta_headers].to_numpy()[0]  # First row since theta is constant per file
        
        # **Extract target data**
        target_data = df[target_label].to_numpy()
        
        # **Extract 'phi' (all columns except theta_headers, target, weights, and metadata)**
        # Exclude metadata columns that shouldn't be features
        metadata_columns = ['fEventID', 'source_file']
        phi_headers = [col for col in df.columns 
                      if col not in theta_headers + target_label + weights_labels + metadata_columns + ['fidelity']]
        phi_data = df[phi_headers].to_numpy()

        # **Extract weights data**
        existing_weights = [w for w in weights_labels if w in df.columns]
        weights_data = df[existing_weights].to_numpy() if existing_weights else np.ones((len(df), 1))
        weights_labels_final = existing_weights if existing_weights else ["weights"]
        
        # **Extract fidelity data (if available)**
        fidelity_data = df[['fidelity']].to_numpy() if 'fidelity' in df else np.ones((len(df), 1))

        # **Store Data in HDF5**
        with h5py.File(hdf5_file, "w") as hdf:
            hdf.create_dataset("fidelity", data=fidelity_data, compression="gzip")
            hdf.create_dataset("theta", data=theta_data, compression="gzip")
            hdf.create_dataset("theta_headers", data=np.array(theta_headers, dtype='S'), compression="gzip")
            hdf.create_dataset("phi", data=phi_data, compression="gzip")
            hdf.create_dataset("phi_labels", data=np.array(phi_headers, dtype='S'), compression="gzip")
            hdf.create_dataset("target", data=target_data, compression="gzip")
            hdf.create_dataset("target_labels", data=np.array(target_label, dtype='S'), compression="gzip")
            hdf.create_dataset("weights", data=weights_data, compression="gzip")
            hdf.create_dataset("weights_labels", data=np.array(weights_labels_final, dtype='S'), compression="gzip")
        
        print(f"  Successfully converted {os.path.basename(csv_file)} -> {os.path.basename(hdf5_file)}")
        
    except Exception as e:
        print(f"  Error converting {csv_file} to HDF5: {e}")

def process_all_csv_files_for_weights(base_directory):
    """
    Process all CSV files in the base directory and its subdirectories to add weights.
    
    Args:
        base_directory (str): Path to the base processed data directory
    """
    # Find all CSV files recursively in the directory
    csv_pattern = os.path.join(base_directory, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {base_directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process for weights:")
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Adding weights to CSV files"):
        print(f"Processing: {csv_file}")
        add_weights_column_to_csv(csv_file)
    
    print(f"\nCompleted processing {len(csv_files)} CSV files for weights.")

def process_all_csv_files_for_hdf5(base_directory):
    """
    Process all CSV files in the base directory and its subdirectories to convert to HDF5.
    
    Args:
        base_directory (str): Path to the base processed data directory
    """
    # Find all CSV files recursively in the directory
    csv_pattern = os.path.join(base_directory, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {base_directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files to convert to HDF5:")
    
    # Define column mappings based on COHERENT data structure
    theta_headers = ['water_shielding_mm', 'veto_thickness_mm']
    target_label = ['veto_active']  # Main prediction target
    weights_labels = ['weights']
    
    print(f"Column mapping:")
    print(f"  Theta (design parameters): {theta_headers}")
    print(f"  Target (prediction): {target_label}")
    print(f"  Weights: {weights_labels}")
    print(f"  Phi (features): All other columns except metadata\n")
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Converting CSV files to HDF5"):
        print(f"Processing: {csv_file}")
        
        # Create HDF5 filename in the same directory
        hdf5_file = csv_file.replace(".csv", ".h5")
        
        # Skip if HDF5 file already exists
        if os.path.exists(hdf5_file):
            print(f"  HDF5 file already exists: {os.path.basename(hdf5_file)}")
            continue
            
        # Convert the file
        convert_single_csv_to_hdf5(csv_file, hdf5_file, theta_headers, target_label, weights_labels)
    
    print(f"\nCompleted converting {len(csv_files)} CSV files to HDF5.")

def process_theta_headers():
    """Process CSV files to add theta headers (original functionality)"""
    # Get base data directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = current_script_dir  # This is the data directory
    
    # Process all veto CSV files in lf and hf directories
    input_directories = [
        os.path.join(base_data_dir, "lf"),
        os.path.join(base_data_dir, "hf")
    ]
    
    total_processed = 0
    for input_directory in input_directories:
        if os.path.exists(input_directory):
            print(f"\nProcessing files in: {input_directory}")
            veto_files = glob.glob(os.path.join(input_directory, "*Veto*.csv"))
            veto_files.extend(glob.glob(os.path.join(input_directory, "*baseline*.csv")))
            
            for veto_file in veto_files:
                try:
                    # Test config extraction
                    config = extract_veto_config_from_filename(veto_file)
                    print(f"Detected config '{config}' for file: {os.path.basename(veto_file)}")
                    
                    # Process files if config is valid
                    if config and config in veto_config_mapping:
                        add_theta_headers_to_csv(veto_file, base_data_dir)
                        total_processed += 1
                    else:
                        print(f"Skipping {os.path.basename(veto_file)} - no valid config mapping")
                except Exception as e:
                    print(f"Error processing {veto_file}: {e}")
            
            print(f"Found {len(veto_files)} files in {input_directory}")
        else:
            print(f"Directory does not exist: {input_directory}")
    
    print(f"\nTotal files processed: {total_processed}")
    print(f"Files saved to: {os.path.join(base_data_dir, 'processed', 'lf')} and {os.path.join(base_data_dir, 'processed', 'hf')}")

def process_weights():
    """Process CSV files to add weights column"""
    # Define the base directory path for processed files
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_directory = os.path.join(current_script_dir, "processed")
    
    # Check if the directory exists
    if not os.path.exists(processed_directory):
        print(f"Error: Directory {processed_directory} does not exist!")
        return
    
    print(f"Processing CSV files in: {processed_directory}")
    print("This will add a 'weights' column with value 1 to all CSV files in place.\n")
    
    # Process all CSV files
    process_all_csv_files_for_weights(processed_directory)

def process_hdf5_conversion():
    """Process CSV files to convert them to HDF5 format"""
    # Define the base directory path for processed files
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_directory = os.path.join(current_script_dir, "processed")
    
    # Check if the directory exists
    if not os.path.exists(processed_directory):
        print(f"Error: Directory {processed_directory} does not exist!")
        return
    
    print(f"Converting CSV files to HDF5 in: {processed_directory}")
    print("This will create .h5 files alongside the existing .csv files.\n")
    
    # Process all CSV files
    process_all_csv_files_for_hdf5(processed_directory)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Data preprocessing tool for COHERENT experiment data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocessing.py --add-theta       # Add theta headers to CSV files
  python preprocessing.py --add-weights     # Add weights column to processed CSV files
  python preprocessing.py --convert-hdf5    # Convert processed CSV files to HDF5 format
  python preprocessing.py --help            # Show this help message
        """
    )
    
    # Add mutually exclusive group for different operations
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--add-theta',
        action='store_true',
        help='Add theta headers (water_shielding_mm, veto_thickness_mm) to CSV files from lf/ and hf/ directories'
    )
    group.add_argument(
        '--add-weights',
        action='store_true',
        help='Add weights column with value 1 to all CSV files in the processed/ directory'
    )
    group.add_argument(
        '--convert-hdf5',
        action='store_true',
        help='Convert all CSV files in processed/ directory to HDF5 format'
    )
    
    args = parser.parse_args()
    
    # Execute the appropriate function based on arguments
    if args.add_theta:
        print("=" * 60)
        print("ADDING THETA HEADERS TO CSV FILES")
        print("=" * 60)
        process_theta_headers()
    elif args.add_weights:
        print("=" * 60)
        print("ADDING WEIGHTS COLUMN TO CSV FILES")
        print("=" * 60)
        process_weights()
    elif args.convert_hdf5:
        print("=" * 60)
        print("CONVERTING CSV FILES TO HDF5 FORMAT")
        print("=" * 60)
        process_hdf5_conversion()

if __name__ == "__main__":
    main()
