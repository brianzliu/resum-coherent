#!/usr/bin/env python3
"""
Example usage of the MFGP Prediction Pipeline

This script demonstrates how to use the MFGP prediction pipeline
with sample data files.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from mfgp_prediction_pipeline import MFGPPredictor

def create_sample_input_files():
    """Create sample input files for demonstration"""
    
    # Create sample directory
    sample_dir = Path("sample_input_files")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample file 1: Random theta values with known y_raw values (for coverage testing)
    np.random.seed(42)
    n_points_1 = 50
    
    # Generate theta values within reasonable ranges
    water_shielding = np.random.uniform(40, 120, n_points_1)
    veto_thickness = np.random.uniform(10, 90, n_points_1)
    
    # Generate synthetic y_raw values (for demonstration)
    y_raw_1 = 0.1 + 0.001 * water_shielding - 0.0005 * veto_thickness + np.random.normal(0, 0.01, n_points_1)
    
    sample_data_1 = pd.DataFrame({
        'water_shielding_mm': water_shielding,
        'veto_thickness_mm': veto_thickness, 
        'y_raw': y_raw_1
    })
    
    file1_path = sample_dir / "sample_data_with_yraw.csv"
    sample_data_1.to_csv(file1_path, index=False)
    
    # Sample file 2: Specific theta values without y_raw (prediction only)
    theta_values_2 = np.array([
        [50.0, 25.0],
        [75.0, 50.0], 
        [100.0, 75.0],
        [116.2, 10.0],  # Same as one of the test points from notebook
        [91.2, 35.0],   # Same as one of the test points from notebook
        [41.2, 85.0]    # Same as one of the test points from notebook
    ])
    
    sample_data_2 = pd.DataFrame({
        'water_shielding_mm': theta_values_2[:, 0],
        'veto_thickness_mm': theta_values_2[:, 1]
    })
    
    file2_path = sample_dir / "sample_theta_values.csv"
    sample_data_2.to_csv(file2_path, index=False)
    
    # Sample file 3: Grid of theta values for contour overlay
    water_grid = np.linspace(45, 115, 8)
    veto_grid = np.linspace(15, 85, 8)
    
    grid_points = []
    for w in water_grid:
        for v in veto_grid:
            grid_points.append([w, v])
    
    sample_data_3 = pd.DataFrame({
        'water_shielding_mm': [p[0] for p in grid_points],
        'veto_thickness_mm': [p[1] for p in grid_points]
    })
    
    file3_path = sample_dir / "grid_theta_values.csv"  
    sample_data_3.to_csv(file3_path, index=False)
    
    print(f"✓ Created sample input files:")
    print(f"  - {file1_path} ({len(sample_data_1)} points with y_raw)")
    print(f"  - {file2_path} ({len(sample_data_2)} specific theta values)")  
    print(f"  - {file3_path} ({len(sample_data_3)} grid points)")
    
    return [str(file1_path), str(file2_path), str(file3_path)]

def example_usage():
    """Demonstrate the prediction pipeline usage"""
    
    print("MFGP Prediction Pipeline - Example Usage")
    print("="*50)
    
    # Step 1: Create sample input files
    print("\n1. Creating sample input files...")
    input_files = create_sample_input_files()
    
    # Step 2: Specify model path (adjust this to your actual model path)
    model_path = "../coherent/out/mfgp/saved_models"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at {model_path}")
        print("Please run the sonata_mfgp.ipynb notebook first to train and save the model.")
        print("The model should be saved in the 'saved_models' subdirectory.")
        return
    
    # Step 3: Initialize the predictor
    print(f"\n2. Loading MFGP model from {model_path}...")
    try:
        predictor = MFGPPredictor(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Step 4: Run the complete pipeline
    print(f"\n3. Running prediction pipeline...")
    output_dir = "example_predictions_output"
    
    try:
        predictor.run_complete_pipeline(input_files, output_dir)
        print(f"\n✅ Example completed successfully!")
        print(f"Check the '{output_dir}' directory for results.")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return

def command_line_example():
    """Show how to use the pipeline from command line"""
    
    print("\n" + "="*60)
    print("COMMAND LINE USAGE EXAMPLE")
    print("="*60)
    
    print("\nAfter creating your input files, you can run the pipeline from command line:")
    print("\n# Basic usage:")
    print("python mfgp_prediction_pipeline.py \\")
    print("  --model_path ../coherent/out/mfgp/saved_models \\")
    print("  --input_files file1.csv file2.csv file3.csv")
    
    print("\n# With custom output directory:")
    print("python mfgp_prediction_pipeline.py \\")
    print("  --model_path ../coherent/out/mfgp/saved_models \\")
    print("  --input_files file1.csv file2.csv \\")
    print("  --output_dir my_predictions")
    
    print("\n# With custom config file:")
    print("python mfgp_prediction_pipeline.py \\")
    print("  --model_path ../coherent/out/mfgp/saved_models \\")
    print("  --input_files file1.csv file2.csv \\")
    print("  --config_path ../coherent/my_settings.yaml")

def quick_prediction_example():
    """Show how to make quick predictions programmatically"""
    
    print("\n" + "="*60) 
    print("PROGRAMMATIC USAGE EXAMPLE")
    print("="*60)
    
    model_path = "../coherent/out/mfgp/saved_models"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    try:
        # Initialize predictor
        predictor = MFGPPredictor(model_path)
        
        # Define some theta values for prediction
        theta_values = np.array([
            [116.2, 10.0],   # Test point from notebook
            [91.2, 35.0],    # Test point from notebook  
            [41.2, 85.0],    # Test point from notebook
            [80.0, 40.0],    # New point
            [60.0, 60.0]     # New point
        ])
        
        # Make predictions
        print(f"\nMaking predictions for {len(theta_values)} theta values...")
        predictions, uncertainties = predictor.predict_y_raw(theta_values)
        
        # Display results
        print(f"\nPrediction Results:")
        print(f"{'Theta Values':<20} {'Prediction':<12} {'Uncertainty':<12}")
        print("-" * 45)
        
        for i, (theta, pred, unc) in enumerate(zip(theta_values, predictions, uncertainties)):
            theta_str = f"({theta[0]:.1f}, {theta[1]:.1f})"
            print(f"{theta_str:<20} {pred:.6f}    ±{unc:.6f}")
        
        return predictions, uncertainties
        
    except Exception as e:
        print(f"❌ Quick prediction failed: {e}")
        return None, None

if __name__ == "__main__":
    # Run the complete example
    example_usage()
    
    # Show command line usage
    command_line_example()
    
    # Show programmatic usage
    quick_prediction_example() 