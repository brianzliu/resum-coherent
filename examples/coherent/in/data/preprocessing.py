import pandas as pd
import os
import glob
import re

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


if __name__ == "__main__":
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
