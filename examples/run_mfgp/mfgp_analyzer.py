"""
Multi-Fidelity Gaussian Process Analyzer

A comprehensive automated analysis pipeline for multi-fidelity Gaussian process predictions.
Handles CSV file processing, prediction generation, uncertainty visualization, and coverage analysis.

Author: Generated for MFGP analysis pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from collections import defaultdict
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist

# Import required modules for MFGP
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays


class MFGPAnalyzer:
    """
    Automated analysis pipeline for multi-fidelity Gaussian process predictions.
    
    This class provides comprehensive tools for:
    - Processing multiple CSV files with theta combinations
    - Generating predictions with uncertainty quantification
    - Creating uncertainty band visualizations
    - Calculating coverage statistics
    - Generating contour maps and summary plots
    """
    
    def __init__(self, mf_model, x_labels, y_label_sim='y_raw', output_dir=None):
        """
        Initialize the MFGPAnalyzer.
        
        Parameters:
        -----------
        mf_model : GPyMultiOutputWrapper
            The trained multi-fidelity Gaussian process model
        x_labels : list
            List of parameter names (e.g., ['water_shielding_mm', 'veto_thickness_mm'])
        y_label_sim : str
            Name of the target variable column (default: 'y_raw')
        output_dir : str or Path
            Directory to save output plots and results (default: current directory)
        """
        self.mf_model = mf_model
        self.x_labels = x_labels
        self.y_label_sim = y_label_sim
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"MFGPAnalyzer initialized:")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Parameter labels: {self.x_labels}")
        print(f"  Target variable: {self.y_label_sim}")
        
    def load_and_process_csv_files(self, file_patterns, fidelity_filter=1.0, iteration_filter=0):
        """
        Load and process multiple CSV files to extract unique theta combinations.
        
        Parameters:
        -----------
        file_patterns : list or str
            List of file patterns or single pattern to match CSV files
            Supports glob patterns like 'data/*.csv' or specific file paths
        fidelity_filter : float
            Fidelity level to filter (default: 1.0 for high fidelity)
        iteration_filter : int
            Iteration number to filter (default: 0)
            
        Returns:
        --------
        dict: Dictionary with file names as keys and processed data as values
        """
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]
            
        all_files = []
        for pattern in file_patterns:
            matched_files = glob.glob(pattern)
            all_files.extend(matched_files)
            
        if not all_files:
            print(f"Warning: No files found matching patterns: {file_patterns}")
            return {}
            
        processed_data = {}
        
        for file_path in all_files:
            print(f"Processing: {file_path}")
            try:
                df = pd.read_csv(file_path)
                
                # Check if required columns exist
                required_cols = self.x_labels + [self.y_label_sim, 'fidelity', 'iteration']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"  Warning: Missing columns {missing_cols}, skipping file")
                    continue
                
                # Filter data
                filtered_df = df[(df['fidelity'] == fidelity_filter) & 
                               (df['iteration'] == iteration_filter)]
                
                if filtered_df.empty:
                    print(f"  Warning: No data found with fidelity={fidelity_filter}, iteration={iteration_filter}")
                    continue
                
                # Group by unique theta combinations
                theta_groups = {}
                unique_combinations = filtered_df[self.x_labels].drop_duplicates()
                
                for _, combo in unique_combinations.iterrows():
                    combo_key = tuple(combo.values)
                    mask = True
                    for i, label in enumerate(self.x_labels):
                        mask &= (filtered_df[label] == combo.iloc[i])
                    
                    group_data = filtered_df[mask]
                    theta_groups[combo_key] = {
                        'x_values': combo.values,
                        'y_values': group_data[self.y_label_sim].values,
                        'raw_data': group_data
                    }
                
                processed_data[Path(file_path).name] = {
                    'theta_groups': theta_groups,
                    'full_data': filtered_df,
                    'file_path': file_path
                }
                
                print(f"  Found {len(theta_groups)} unique theta combinations")
                
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                
        return processed_data
    
    def predict_for_theta_groups(self, processed_data):
        """
        Generate predictions for all theta combinations in processed data.
        
        Parameters:
        -----------
        processed_data : dict
            Dictionary returned by load_and_process_csv_files()
            
        Returns:
        --------
        dict: Nested dictionary with predictions for each file and theta combination
        """
        predictions = {}
        
        for file_name, file_data in processed_data.items():
            print(f"\nGenerating predictions for {file_name}")
            file_predictions = {}
            
            for combo_key, group_data in file_data['theta_groups'].items():
                # Prepare prediction input (add fidelity=1 for high-fidelity prediction)
                x_pred = np.array([list(combo_key) + [1.0]])  # Add fidelity indicator
                
                # Get model predictions
                mean_pred, var_pred = self.mf_model.predict(x_pred)
                std_pred = np.sqrt(var_pred)
                
                file_predictions[combo_key] = {
                    'x_values': group_data['x_values'],
                    'y_true': group_data['y_values'],
                    'y_pred_mean': mean_pred[0, 0],
                    'y_pred_std': std_pred[0, 0],
                    'raw_data': group_data['raw_data']
                }
                
                print(f"  Theta {combo_key}: mean={mean_pred[0, 0]:.6f}, std={std_pred[0, 0]:.6f}")
            
            predictions[file_name] = file_predictions
            
        return predictions
    
    def calculate_coverage_statistics(self, predictions):
        """
        Calculate coverage statistics for all predictions.
        
        Parameters:
        -----------
        predictions : dict
            Dictionary returned by predict_for_theta_groups()
            
        Returns:
        --------
        dict: Coverage statistics for each file
        """
        coverage_stats = {}
        
        for file_name, file_preds in predictions.items():
            print(f"\nCalculating coverage for {file_name}")
            
            all_deviations = []
            sigma_bands = [1, 2, 3]
            coverage_counts = {sigma: 0 for sigma in sigma_bands}
            total_points = 0
            
            for combo_key, pred_data in file_preds.items():
                y_true = pred_data['y_true']
                y_pred_mean = pred_data['y_pred_mean']
                y_pred_std = pred_data['y_pred_std']
                
                # Calculate deviations for each data point
                deviations = np.abs(y_true - y_pred_mean) / y_pred_std
                all_deviations.extend(deviations)
                
                # Count coverage for different sigma bands
                for sigma in sigma_bands:
                    coverage_counts[sigma] += np.sum(deviations <= sigma)
                
                total_points += len(y_true)
            # Calculate percentages
            coverage_percentages = {sigma: 100 * count / total_points 
                                  for sigma, count in coverage_counts.items()}
            
            coverage_stats[file_name] = {
                'coverage_counts': coverage_counts,
                'coverage_percentages': coverage_percentages,
                'total_points': total_points,
                'all_deviations': np.array(all_deviations)
            }
            
            print(f"  Total points: {total_points}")
            for sigma in sigma_bands:
                print(f"  ±{sigma}σ: {coverage_counts[sigma]}/{total_points} ({coverage_percentages[sigma]:.1f}%)")
                
        return coverage_stats

    def plot_uncertainty_bands_for_theta_group(self, combo_key, pred_data, file_name, save_plots=True):
        """
        Plot uncertainty bands for a specific theta combination with coverage statistics.
        
        Parameters:
        -----------
        combo_key : tuple
            Theta combination values
        pred_data : dict
            Prediction data for this theta combination
        file_name : str
            Name of the source file
        save_plots : bool
            Whether to save the plot to disk
        """
        y_true = pred_data['y_true']
        y_pred_mean = pred_data['y_pred_mean']
        y_pred_std = pred_data['y_pred_std']
        
        # Create index for plotting
        idx = np.arange(len(y_true))
        
        plt.figure(figsize=(12, 4))
        
        # Plot uncertainty bands
        plt.fill_between(idx, y_pred_mean - 3*y_pred_std, y_pred_mean + 3*y_pred_std, 
                        facecolor='r', alpha=0.1, label='±3σ')
        plt.fill_between(idx, y_pred_mean - 2*y_pred_std, y_pred_mean + 2*y_pred_std, 
                        facecolor='y', alpha=0.15, label='±2σ')
        plt.fill_between(idx, y_pred_mean - 1*y_pred_std, y_pred_mean + 1*y_pred_std, 
                        facecolor='g', alpha=0.2, label='RESuM ±1σ')
        
        # Plot actual data points
        plt.scatter(idx, y_true, color='k', s=20, label='Validation Data', zorder=5)
        
        # Plot mean prediction line
        plt.axhline(y=y_pred_mean, color='red', linestyle='--', alpha=0.8, label='RESuM Mean')
        
        plt.xlabel('Sample Index')
        plt.ylabel(f'{self.y_label_sim}')
        plt.title(f'Theta: {combo_key} | File: {file_name}')
        
        # Custom legend order
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [4, 3, 2, 1, 0]  # Reorder to put validation data first
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
                  ncol=5, loc='upper right', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        
        # Calculate and add coverage statistics as text box
        deviations = np.abs(y_true - y_pred_mean) / y_pred_std
        coverage_text = []
        for sigma in [1, 2, 3]:
            within_sigma = np.sum(deviations <= sigma)
            percentage = 100 * within_sigma / len(y_true)
            coverage_text.append(f"±{sigma}σ: {within_sigma}/{len(y_true)} ({percentage:.1f}%)")
        
        # Add coverage statistics as text box
        text_str = "Coverage Statistics:\n" + "\n".join(coverage_text)
        plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        if save_plots:
            combo_str = '_'.join([f'{val:.1f}' for val in combo_key])
            filename = f'uncertainty_bands_{Path(file_name).stem}_theta_{combo_str}.png'
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {save_path}")
            
        plt.show()
        
        # Also print coverage statistics to console
        print(f"    Coverage statistics for Theta {combo_key}:")
        for sigma in [1, 2, 3]:
            within_sigma = np.sum(deviations <= sigma)
            percentage = 100 * within_sigma / len(y_true)
            print(f"    ±{sigma}σ: {within_sigma}/{len(y_true)} ({percentage:.1f}%)")
            
    def plot_coverage_summary(self, coverage_stats, save_plots=True):
        """
        Create summary plots of coverage statistics across all files.
        
        Parameters:
        -----------
        coverage_stats : dict
            Dictionary returned by calculate_coverage_statistics()
        save_plots : bool
            Whether to save the plot to disk
        """
        # Collect data for plotting
        files = list(coverage_stats.keys())
        sigma_levels = [1, 2, 3]
        
        # Coverage percentages plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of coverage percentages
        x_pos = np.arange(len(files))
        width = 0.25
        
        for i, sigma in enumerate(sigma_levels):
            percentages = [coverage_stats[f]['coverage_percentages'][sigma] for f in files]
            ax1.bar(x_pos + i*width, percentages, width, 
                   label=f'±{sigma}σ', alpha=0.8)
        
        ax1.set_xlabel('Files')
        ax1.set_ylabel('Coverage Percentage (%)')
        ax1.set_title('Coverage Statistics by File')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels([Path(f).stem for f in files], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add theoretical lines for normal distribution
        theoretical_coverage = {1: 68.27, 2: 95.45, 3: 99.73}
        for sigma, theoretical in theoretical_coverage.items():
            ax1.axhline(y=theoretical, color='gray', linestyle='--', alpha=0.7)
            ax1.text(len(files)-0.5, theoretical+1, f'{theoretical:.1f}% (theory)', 
                    fontsize=9, alpha=0.7)
        
        # Histogram of all deviations
        all_deviations = []
        for stats in coverage_stats.values():
            all_deviations.extend(stats['all_deviations'])
        
        ax2.hist(all_deviations, bins=50, alpha=0.7, density=True, 
                label='Observed Deviations')
        
        # Overlay theoretical normal distribution
        x_theory = np.linspace(0, max(all_deviations), 100)
        y_theory = 2 * np.exp(-0.5 * x_theory**2) / np.sqrt(2*np.pi)  # Half-normal (absolute values)
        ax2.plot(x_theory, y_theory, 'r--', label='Theoretical (Half-Normal)', linewidth=2)
        
        ax2.set_xlabel('|Deviation| (in σ units)')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Prediction Deviations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines for 1, 2, 3 sigma
        for sigma in [1, 2, 3]:
            ax2.axvline(x=sigma, color='gray', linestyle=':', alpha=0.7)
            ax2.text(sigma, ax2.get_ylim()[1]*0.9, f'{sigma}σ', 
                    rotation=90, verticalalignment='top', fontsize=9)
        
        plt.tight_layout()
        
        if save_plots:
            save_path = self.output_dir / 'coverage_summary.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Coverage summary saved: {save_path}")
            
        plt.show()
        
        # Print overall statistics
        total_points = sum(stats['total_points'] for stats in coverage_stats.values())
        print(f"\nOverall Statistics ({len(files)} files, {total_points} total points):")
        
        for sigma in sigma_levels:
            total_within = sum(stats['coverage_counts'][sigma] for stats in coverage_stats.values())
            percentage = 100 * total_within / total_points
            theoretical = theoretical_coverage[sigma]
            print(f"±{sigma}σ: {total_within}/{total_points} ({percentage:.1f}%) | " +
                  f"Theoretical: {theoretical:.1f}% | Diff: {percentage-theoretical:+.1f}%")

    def create_enhanced_contour_plots(self, processed_data, grid_steps=50, levels=25, save_plots=True):
        """
        Create enhanced contour plots showing mean prediction and uncertainty with training data overlaid.
        
        Parameters:
        -----------
        processed_data : dict
            Dictionary returned by load_and_process_csv_files()
        grid_steps : int
            Number of grid points for contour generation
        levels : int
            Number of contour levels
        save_plots : bool
            Whether to save plots to disk
            
        Returns:
        --------
        matplotlib.figure.Figure: The generated figure
        """
        # Collect all data points from all files
        all_x_data = []
        all_y_data = []
        
        for file_data in processed_data.values():
            for combo_key, group_data in file_data['theta_groups'].items():
                all_x_data.extend([[combo_key[0], combo_key[1]]] * len(group_data['y_values']))
                all_y_data.extend(group_data['y_values'])
        
        all_x_data = np.array(all_x_data)
        
        # Get parameter ranges
        param_x_min, param_x_max = all_x_data[:, 1].min(), all_x_data[:, 1].max()  # veto_thickness_mm
        param_y_min, param_y_max = all_x_data[:, 0].min(), all_x_data[:, 0].max()  # water_shielding_mm
        
        # Create prediction grid
        x_vals = np.linspace(param_x_min, param_x_max, grid_steps)
        y_vals = np.linspace(param_y_min, param_y_max, grid_steps)
        Xg, Yg = np.meshgrid(x_vals, y_vals)
        
        # Prepare points for prediction
        points = []
        for y in y_vals:
            for x in x_vals:
                points.append([y, x])  # [water_shielding_mm, veto_thickness_mm]
        
        points = np.array(points)
        fidelity_col = np.ones((len(points), 1))
        points_with_fidelity = np.hstack([points, fidelity_col])
        
        # Get predictions
        mean_pred, var_pred = self.mf_model.predict(points_with_fidelity)
        std_pred = np.sqrt(var_pred)
        
        Z_mean = mean_pred.reshape(grid_steps, grid_steps)
        Z_std = std_pred.reshape(grid_steps, grid_steps)
        
        # Create enhanced plots - only 2 plots now
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        
        # Plot 1: Mean prediction
        contour1 = ax1.contourf(Xg, Yg, Z_mean, levels=levels, cmap='viridis')
        cbar1 = fig.colorbar(contour1, ax=ax1)
        cbar1.set_label(r"Predicted $y_{raw}$ (mean)", fontsize=12)
        
        # Add contour lines
        ax1.contour(Xg, Yg, Z_mean, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # Overlay data from all files
        for file_name, file_data in processed_data.items():
            x_coords = []
            y_coords = []
            for combo_key in file_data['theta_groups'].keys():
                x_coords.append(combo_key[1])  # veto_thickness_mm
                y_coords.append(combo_key[0])  # water_shielding_mm
            
            # Use red for LF data points
            ax1.scatter(x_coords, y_coords, c='red', s=100, 
                       marker='o', edgecolors='black', linewidth=2, 
                       label='LF Data', alpha=0.9, zorder=5)
        
        ax1.set_xlabel(self.x_labels[1], fontsize=12)  # veto_thickness_mm
        ax1.set_ylabel(self.x_labels[0], fontsize=12)  # water_shielding_mm
        ax1.set_title('Mean Prediction', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty map
        contour2 = ax2.contourf(Xg, Yg, Z_std, levels=levels, cmap='Reds')
        cbar2 = fig.colorbar(contour2, ax=ax2)
        cbar2.set_label(r"Uncertainty ($\sigma$)", fontsize=12)
        
        # Add contour lines
        ax2.contour(Xg, Yg, Z_std, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # Overlay data from all files
        for file_name, file_data in processed_data.items():
            x_coords = []
            y_coords = []
            for combo_key in file_data['theta_groups'].keys():
                x_coords.append(combo_key[1])  # veto_thickness_mm
                y_coords.append(combo_key[0])  # water_shielding_mm
            
            # Use red for LF data points
            ax2.scatter(x_coords, y_coords, c='red', s=100, 
                       marker='o', edgecolors='black', linewidth=2, 
                       label='LF Data', alpha=0.9, zorder=5)
        
        ax2.set_xlabel(self.x_labels[1], fontsize=12)
        ax2.set_ylabel(self.x_labels[0], fontsize=12)
        ax2.set_title('Prediction Uncertainty', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        if save_plots:
            save_path = self.output_dir / 'enhanced_contour_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced contour analysis saved: {save_path}")
            
        plt.show()
        
        return fig
    
    def plot_prediction_vs_true(self, predictions, file_name=None, save_plot=True):
        """
        Create a scatter plot of predicted vs true values aggregated per unique theta.

        This updated version first aggregates (averages) the true observed values for each
        unique theta configuration (matching the averaging approach used in
        plot_uncertainty_bands_across_thetas) and then compares those per-theta
        averages against the model's predicted mean with corresponding predictive
        standard deviation as error bars.

        Parameters:
        -----------
        predictions : dict
            Dictionary returned by predict_for_theta_groups()
        file_name : str, optional
            Specific file to plot (default: first file)
        save_plot : bool
            Whether to save the plot to disk
        """
        if file_name is None:
            file_name = list(predictions.keys())[0]
        
        file_preds = predictions[file_name]
        
        # Aggregate per theta: compute mean of true values for each theta
        all_true_mean = []      # per-theta averaged true value
        all_pred_mean = []      # model predicted mean (already per-theta)
        all_pred_std = []       # model predicted std (already per-theta)
        theta_labels = []       # string labels for optional future use / debugging
        
        for combo_key, pred_data in file_preds.items():
            y_true_vals = pred_data['y_true']
            y_true_avg = np.mean(y_true_vals)
            all_true_mean.append(y_true_avg)
            all_pred_mean.append(pred_data['y_pred_mean'])
            all_pred_std.append(pred_data['y_pred_std'])
            theta_labels.append(str(combo_key))
        
        all_true_mean = np.array(all_true_mean)
        all_pred_mean = np.array(all_pred_mean)
        all_pred_std = np.array(all_pred_std)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Scatter with error bars (prediction uncertainty on y-axis)
        plt.errorbar(all_true_mean, all_pred_mean, yerr=all_pred_std,
                     fmt='o', alpha=0.8, markersize=6, capsize=3, label='Per-Theta Mean')
        
        # Perfect prediction line
        min_val = min(all_true_mean.min(), all_pred_mean.min())
        max_val = max(all_true_mean.max(), all_pred_mean.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--',
                 label='Perfect Prediction', linewidth=2)
        
        # Global average std for bands (gives visual sense of typical uncertainty)
        avg_sigma = np.mean(all_pred_std)
        x_line = np.linspace(min_val, max_val, 200)
        for i, sigma_mult in enumerate([1, 2, 3]):
            alpha_val = 0.25 - i * 0.07
            plt.fill_between(x_line, x_line - sigma_mult * avg_sigma,
                              x_line + sigma_mult * avg_sigma,
                              alpha=alpha_val, label=f'±{sigma_mult}σ (avg)')
        
        plt.xlabel(f'True {self.y_label_sim} (per-theta mean)')
        plt.ylabel(f'Predicted {self.y_label_sim}')
        plt.title(f'Predicted vs True (Per-Theta Aggregated)\nFile: {Path(file_name).stem}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics based on aggregated points
        correlation = np.corrcoef(all_true_mean, all_pred_mean)[0, 1] if len(all_true_mean) > 1 else np.nan
        mae = np.mean(np.abs(all_true_mean - all_pred_mean))
        rmse = np.sqrt(np.mean((all_true_mean - all_pred_mean)**2))
        
        stats_text = (f'Points (unique thetas): {len(all_true_mean)}\n'
                      f'Correlation: {correlation:.3f}\n'
                      f'MAE: {mae:.6f}\nRMSE: {rmse:.6f}')
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        plt.tight_layout()
        
        if save_plot:
            save_path = self.output_dir / f'predicted_vs_true_{Path(file_name).stem}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction vs true (aggregated) plot saved: {save_path}")
        
        plt.show()
    
    def plot_uncertainty_bands_across_thetas(self, predictions, processed_data, file_name=None, save_plot=True):
        """
        Plot uncertainty bands across all theta values for a given file, resembling the screenshot.

        Parameters:
        -----------
        predictions : dict
            Dictionary returned by predict_for_theta_groups().
        processed_data : dict
            Dictionary returned by load_and_process_csv_files().
        file_name : str, optional
            Specific file to plot (default: first file).
        save_plot : bool
            Whether to save the plot to disk.
        """
        if file_name is None:
            file_name = list(predictions.keys())[0]

        file_preds = predictions[file_name]
        file_proc_data = processed_data[file_name]

        # Sort theta combinations to ensure consistent ordering and handle missing pairs
        sorted_thetas = sorted(file_preds.keys())

        y_true_means = []
        y_pred_means = []
        y_pred_stds = []

        for theta in sorted_thetas:
            # Average out the y_raw for the true data points
            y_true_values = file_proc_data['theta_groups'][theta]['y_values']
            y_true_means.append(np.mean(y_true_values))

            # Get prediction data
            pred_data = file_preds[theta]
            y_pred_means.append(pred_data['y_pred_mean'])
            y_pred_stds.append(pred_data['y_pred_std'])

        y_true_means = np.array(y_true_means)
        y_pred_means = np.array(y_pred_means)
        y_pred_stds = np.array(y_pred_stds)

        # Create a simple non-skipping index for plotting
        idx = np.arange(len(sorted_thetas))

        plt.figure(figsize=(12, 4))

        # Plot uncertainty bands using colors from the original plot
        plt.fill_between(idx, y_pred_means - 3 * y_pred_stds, y_pred_means + 3 * y_pred_stds,
                         facecolor='r', alpha=0.1, label='±3σ')
        plt.fill_between(idx, y_pred_means - 2 * y_pred_stds, y_pred_means + 2 * y_pred_stds,
                         facecolor='y', alpha=0.15, label='±2σ')
        plt.fill_between(idx, y_pred_means - 1 * y_pred_stds, y_pred_means + 1 * y_pred_stds,
                         facecolor='g', alpha=0.2, label='RESuM ±1σ')

        # Plot actual data points (averaged y_raw)
        plt.scatter(idx, y_true_means, color='k', s=20, label='HF Validation Data', zorder=5)

        plt.xlabel('Simulation Trial Index')
        plt.ylabel(f'Average $y_{{raw}}$')
        plt.title(f'MFGP Predictions Across Theta Values | File: {Path(file_name).stem}')

        # Custom legend order (inside the chart like plot_uncertainty_bands_for_theta_group)
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Reorder to put HF Validation Data first, then uncertainty bands
        desired_order = ['HF Validation Data', 'RESuM ±1σ', '±2σ', '±3σ']
        order_map = {label: i for i, label in enumerate(desired_order)}
        
        try:
            # Sort handles and labels based on the desired order
            sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: order_map.get(x[1], 99))
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)
            plt.legend(sorted_handles, sorted_labels, ncol=4, loc='upper right', fontsize=10)
        except (ValueError, KeyError):
             # if a label is not found, just use the default legend
            plt.legend(ncol=4, loc='upper right', fontsize=10)

        plt.grid(True, alpha=0.3)
        
        # Set y-axis limits with better scaling - zoom in more to reduce white space
        y_min = min(np.min(y_true_means), np.min(y_pred_means - 3 * y_pred_stds))
        y_max = max(np.max(y_true_means), np.max(y_pred_means + 3 * y_pred_stds))
        y_range = y_max - y_min
        plt.ylim(y_min - 0.05 * y_range, y_max + 0.1 * y_range)  # Small padding, less on bottom
        
        # Calculate coverage statistics based on averaged y_raw values vs uncertainty bands
        coverage_text = []
        for sigma in [1, 2, 3]:
            within_sigma = 0
            for i, theta in enumerate(sorted_thetas):
                y_true_mean = y_true_means[i]
                y_pred_mean = y_pred_means[i]
                y_pred_std = y_pred_stds[i]
                
                # Check if averaged y_raw is within sigma bands
                lower_bound = y_pred_mean - sigma * y_pred_std
                upper_bound = y_pred_mean + sigma * y_pred_std
                
                if lower_bound <= y_true_mean <= upper_bound:
                    within_sigma += 1
            
            total_thetas = len(sorted_thetas)
            percentage = 100 * within_sigma / total_thetas
            coverage_text.append(f"±{sigma}σ: {within_sigma}/{total_thetas} ({percentage:.1f}%)")
        
        # Add coverage statistics as text box
        text_str = "Coverage Statistics:\n" + "\n".join(coverage_text)
        plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=10, family='monospace')
        
        plt.tight_layout()

        if save_plot:
            filename = f'uncertainty_bands_across_thetas_{Path(file_name).stem}.png'
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved plot: {save_path}")

        plt.show()

    def run_complete_analysis(self, file_patterns, fidelity_filter=1.0, iteration_filter=0, 
                            plot_individual_groups=True, save_all_plots=True):
        """
        Run the complete automated analysis pipeline.
        
        Parameters:
        -----------
        file_patterns : list or str
            File patterns to process
        fidelity_filter : float
            Fidelity level to filter
        iteration_filter : int
            Iteration number to filter
        plot_individual_groups : bool
            Whether to plot individual theta combinations
        save_all_plots : bool
            Whether to save all plots to disk
            
        Returns:
        --------
        dict: Complete analysis results
        """
        print("="*80)
        print("AUTOMATED MFGP ANALYSIS PIPELINE")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        
        # Step 1: Load and process CSV files
        print("\n1. Loading and processing CSV files...")
        processed_data = self.load_and_process_csv_files(file_patterns, fidelity_filter, iteration_filter)
        
        if not processed_data:
            print("No data found! Check your file patterns and filters.")
            return None
            
        # Step 2: Generate predictions
        print("\n2. Generating predictions for all theta combinations...")
        predictions = self.predict_for_theta_groups(processed_data)
        
        # Step 3: Calculate coverage statistics
        print("\n3. Calculating coverage statistics...")
        coverage_stats = self.calculate_coverage_statistics(predictions)
        
        # Step 4: Plot individual groups (if requested)
        if plot_individual_groups:
            print("\n4. Generating individual uncertainty band plots...")
            for file_name, file_preds in predictions.items():
                print(f"\nPlotting groups for {file_name}:")
                for combo_key, pred_data in file_preds.items():
                    print(f"  Theta: {combo_key}")
                    self.plot_uncertainty_bands_for_theta_group(
                        combo_key, pred_data, file_name, save_all_plots)
        
        # Step 5: Create coverage summary
        print("\n5. Creating coverage summary plots...")
        self.plot_coverage_summary(coverage_stats, save_all_plots)
        
        # Step 6: Create enhanced contour plots
        print("\n6. Creating enhanced contour analysis...")
        self.create_enhanced_contour_plots(processed_data, save_plots=save_all_plots)
        
        # Step 7: Create prediction vs true plots
        print("\n7. Creating prediction vs true value plots...")
        for file_name in predictions.keys():
            self.plot_prediction_vs_true(predictions, file_name, save_all_plots)
        
        # Step 8: Create plot across all theta values
        print("\n8. Creating plot across all theta values...")
        for file_name in predictions.keys():
            self.plot_uncertainty_bands_across_thetas(predictions, processed_data, file_name, save_all_plots)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        if save_all_plots:
            print(f"All plots saved to: {self.output_dir}")
            
        return {
            'processed_data': processed_data,
            'predictions': predictions,
            'coverage_stats': coverage_stats
        }


def explore_predictions(predictions, file_name=None):
    """
    Utility function to explore prediction results interactively.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary returned by MFGPAnalyzer.predict_for_theta_groups()
    file_name : str, optional
        Specific file to explore (default: first file)
    """
    if file_name is None:
        file_name = list(predictions.keys())[0]
        
    print(f"Exploring predictions for: {file_name}")
    print("-" * 50)
    
    file_preds = predictions[file_name]
    
    for i, (combo_key, pred_data) in enumerate(file_preds.items()):
        print(f"\n{i+1}. Theta: {combo_key}")
        print(f"   Predicted mean: {pred_data['y_pred_mean']:.6f}")
        print(f"   Predicted std:  {pred_data['y_pred_std']:.6f}")
        print(f"   True values:    {len(pred_data['y_true'])} samples")
        print(f"   Value range:    [{np.min(pred_data['y_true']):.6f}, {np.max(pred_data['y_true']):.6f}]")
        
        # Calculate simple metrics
        residuals = pred_data['y_true'] - pred_data['y_pred_mean']
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        normalized_rmse = rmse / pred_data['y_pred_std']
        
        print(f"   MAE:            {mae:.6f}")
        print(f"   RMSE:           {rmse:.6f}")
        print(f"   Normalized RMSE: {normalized_rmse:.3f}σ")
