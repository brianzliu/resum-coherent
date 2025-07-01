#!/usr/bin/env python3
"""
Simple wrapper script to run HDF5 vs CSV comparison with command line arguments.
"""

import sys
import os
import argparse

# Add the current directory to path so we can import our comparison module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Compare HDF5 and CSV file data counts')
    parser.add_argument('--dir', '-d', 
                        default='/home/bliu4/resum-coherent2/examples/coherent/in/data/processed',
                        help='Directory containing HDF5 and CSV files to compare')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output for each file pair')
    
    args = parser.parse_args()
    
    # Import and modify the comparison script
    import compare_hdf5_csv_counts
    
    # Temporarily modify the processed_dir in the main function
    original_main = compare_hdf5_csv_counts.main
    
    def modified_main():
        processed_dir = args.dir
        
        if not os.path.exists(processed_dir):
            print(f"Error: Directory {processed_dir} does not exist!")
            return
        
        print(f"Comparing HDF5 and CSV file data counts in: {processed_dir}")
        print("=" * 80)
        
        file_pairs = compare_hdf5_csv_counts.find_file_pairs(processed_dir)
        
        if not file_pairs:
            print("No HDF5-CSV file pairs found!")
            return
        
        results = []
        
        for hdf5_file, csv_file in file_pairs:
            rel_hdf5 = os.path.relpath(hdf5_file, processed_dir)
            rel_csv = os.path.relpath(csv_file, processed_dir)
            
            hdf5_count = compare_hdf5_csv_counts.count_hdf5_rows(hdf5_file)
            csv_count = compare_hdf5_csv_counts.count_csv_rows(csv_file)
            
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
            
            if args.verbose:
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
    
    # Run the modified version
    modified_main()

if __name__ == "__main__":
    main() 