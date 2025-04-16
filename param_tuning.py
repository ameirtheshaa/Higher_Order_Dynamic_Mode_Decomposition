"""
Parameter Tuning Example for TensorDMD

This example demonstrates how to find optimal parameters for TensorDMD,
including Tucker decomposition ranks and DMD mode truncation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tensor_dmd import TensorDMD
import time
from itertools import product


def main():
    """Run parameter tuning example for TensorDMD."""
    print("Parameter Tuning Example for TensorDMD")
    
    # Create output directory
    os.makedirs('results/param_tuning', exist_ok=True)
    
    # 1. Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    n_time, nx, ny = 40, 30, 30
    test_data = TensorDMD.generate_example_tensor(
        n_time=n_time,
        nx=nx,
        ny=ny,
        freq1=0.15,
        freq2=0.05,
        noise_level=0.05  # Add moderate noise to make tuning more challenging
    )
    print(f"Test data shape: {test_data.shape}")
    
    # 2. Split data into training and testing sets
    train_ratio = 0.7
    train_idx = int(n_time * train_ratio)
    
    train_data = test_data[:train_idx]
    test_data = test_data[train_idx:]
    
    print(f"Training data: {train_data.shape}, Test data: {test_data.shape}")
    
    # 3. Define parameter grid to search
    ranks = [5, 10, 15, 20, None]  # Spatial ranks to try (None = no reduction)
    truncate_ranks = [3, 5, 8, 10, 15]  # DMD mode truncation to try
    
    # 4. Perform grid search
    print("\nStarting parameter grid search...")
    results = []
    
    # Times for prediction
    test_times = np.arange(train_idx, n_time) * 0.1
    
    for r in ranks:
        for tr in truncate_ranks:
            # Skip if truncate_rank > spatial rank (would cause errors)
            if r is not None and tr > r:
                continue
                
            print(f"Testing spatial_rank={r}, truncate_rank={tr}")
            
            try:
                # Record timing
                start_time = time.time()
                
                # Create TensorDMD with these parameters
                tdmd = TensorDMD(
                    train_data,
                    dt=0.1,
                    rank=(None, r, r),  # Apply rank reduction to spatial dimensions only
                    truncate_rank=tr
                )
                
                # Compute error metrics for training data (reconstruction)
                train_reconstructed = tdmd.reconstruct()
                train_error = np.linalg.norm((train_data - train_reconstructed).reshape(-1)) / np.linalg.norm(train_data.reshape(-1))
                
                # Predict test data
                test_predictions = tdmd.predict(test_times)
                
                # Compute error metrics for test data (prediction)
                test_error = np.linalg.norm((test_data - test_predictions).reshape(-1)) / np.linalg.norm(test_data.reshape(-1))
                
                # Compute computation time
                comp_time = time.time() - start_time
                
                # Get eigenvalue info
                spectral_radius = np.max(np.abs(tdmd.Lambda))
                is_stable = spectral_radius <= 1.0
                
                # Store results
                results.append({
                    'spatial_rank': r,
                    'truncate_rank': tr,
                    'train_error': train_error,
                    'test_error': test_error,
                    'computation_time': comp_time,
                    'spectral_radius': spectral_radius,
                    'stable': is_stable
                })
                
            except Exception as e:
                print(f"Error with parameters spatial_rank={r}, truncate_rank={tr}: {e}")
    
    # 5. Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/param_tuning/tuning_results.csv', index=False)
    
    print("\nParameter tuning results:")
    print(results_df)
    
    # 6. Plot results for different parameter combinations
    print("\nCreating result plots...")
    
    # 6.1 Training vs test error
    plt.figure(figsize=(10, 6))
    
    for r in ranks:
        # Filter results for this rank
        r_results = results_df[results_df['spatial_rank'] == r]
        if len(r_results) > 0:
            label = f"Spatial Rank: {r if r is not None else 'None'}"
            plt.plot(r_results['truncate_rank'], r_results['train_error'], 'o-', label=label)
    
    plt.xlabel('Truncation Rank')
    plt.ylabel('Training Error')
    plt.title('Training Error vs Truncation Rank')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results/param_tuning/training_error.png', dpi=300)
    
    # 6.2 Test error
    plt.figure(figsize=(10, 6))
    
    for r in ranks:
        # Filter results for this rank
        r_results = results_df[results_df['spatial_rank'] == r]
        if len(r_results) > 0:
            label = f"Spatial Rank: {r if r is not None else 'None'}"
            plt.plot(r_results['truncate_rank'], r_results['test_error'], 'o-', label=label)
    
    plt.xlabel('Truncation Rank')
    plt.ylabel('Test Error')
    plt.title('Test Error vs Truncation Rank')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results/param_tuning/test_error.png', dpi=300)
    
    # 6.3 Computation time
    plt.figure(figsize=(10, 6))
    
    for r in ranks:
        # Filter results for this rank
        r_results = results_df[results_df['spatial_rank'] == r]
        if len(r_results) > 0:
            label = f"Spatial Rank: {r if r is not None else 'None'}"
            plt.plot(r_results['truncate_rank'], r_results['computation_time'], 'o-', label=label)
    
    plt.xlabel('Truncation Rank')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time vs Truncation Rank')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results/param_tuning/computation_time.png', dpi=300)
    
    # 6.4 Error tradeoff plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['train_error'], results_df['test_error'], 
                c=results_df['truncate_rank'], cmap='viridis', 
                s=80, alpha=0.7)
    
    # Add text labels for each point
    for i, row in results_df.iterrows():
        plt.text(row['train_error'], row['test_error'], 
                 f"({row['spatial_rank']},{row['truncate_rank']})", 
                 fontsize=8)
    
    plt.colorbar(label='Truncation Rank')
    plt.xlabel('Training Error')
    plt.ylabel('Test Error')
    plt.title('Error Tradeoff: Training vs Test')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/param_tuning/error_tradeoff.png', dpi=300)
    
    # 7. Identify optimal parameters
    # Sort by test error (primary) and computation time (secondary)
    best_models = results_df.sort_values(by=['test_error', 'computation_time'])
    top_3 = best_models.head(3)
    
    print("\nTop 3 parameter combinations:")
    print(top_3)
    
    # 8. Create TensorDMD instance with best parameters
    print("\nCreating model with optimal parameters...")
    best_params = top_3.iloc[0]
    best_spatial_rank = best_params['spatial_rank']
    best_truncate_rank = best_params['truncate_rank']
    
    print(f"Best parameters: spatial_rank={best_spatial_rank}, truncate_rank={best_truncate_rank}")
    
    # Create TensorDMD with optimal parameters on full dataset
    optimal_tdmd = TensorDMD(
        test_data,  # Use full dataset
        dt=0.1,
        rank=(None, best_spatial_rank, best_spatial_rank),
        truncate_rank=best_truncate_rank
    )
    
    # 9. Show summary information for optimal model
    mode_info = optimal_tdmd.mode_significance()
    print("\nSignificant modes in optimal model:")
    print(mode_info.head())
    
    # 10. Save diagnostic report for optimal model
    report = optimal_tdmd.generate_diagnostic_report()
    with open('results/param_tuning/optimal_model_report.md', 'w') as f:
        f.write(report)
    
    # 11. Plot spectrum and modes for optimal model
    fig = optimal_tdmd.plot_spectrum()
    plt.savefig('results/param_tuning/optimal_spectrum.png', dpi=300)
    
    fig = optimal_tdmd.visualize_modes(n_modes=3)
    plt.savefig('results/param_tuning/optimal_modes.png', dpi=300)
    
    print("\nParameter tuning completed. Results saved to 'results/param_tuning/'")


if __name__ == "__main__":
    main()