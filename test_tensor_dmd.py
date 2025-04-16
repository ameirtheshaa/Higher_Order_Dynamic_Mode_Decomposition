#!/usr/bin/env python3
"""
Test script for TensorDMD class.

This script demonstrates the basic functionality of the TensorDMD class
by generating a synthetic dataset with known oscillatory patterns and
analyzing it using tensor-based DMD.

Usage:
    python test_tensor_dmd.py

"""

import numpy as np
import matplotlib.pyplot as plt
from tensor_dmd import TensorDMD
import os

# Create output directory
os.makedirs('results', exist_ok=True)


def main():
    """Run a test of the TensorDMD class with a synthetic dataset."""
    print("Testing TensorDMD with synthetic data...")
    
    # 1. Generate synthetic tensor data with known oscillatory patterns
    print("\nGenerating synthetic data...")
    n_time, nx, ny = 80, 30, 30
    tensor_data = TensorDMD.generate_example_tensor(
        n_time=n_time, 
        nx=nx, 
        ny=ny, 
        freq1=0.2,  # First frequency
        freq2=0.05,  # Second frequency
        noise_level=0.02  # Noise level
    )
    print(f"Data shape: {tensor_data.shape}")
    
    # 2. Create TensorDMD instance with reduced rank for spatial dimensions
    print("\nInitializing TensorDMD with rank reduction...")
    tdmd = TensorDMD(
        tensor_data, 
        dt=0.1, 
        rank=(None, 15, 15),  # Reduce spatial dimensions, keep time dimension
        truncate_rank=10  # Keep only top 10 DMD modes
    )
    
    # 3. Analyze the DMD modes and their properties
    print("\nAnalyzing DMD modes...")
    mode_info = tdmd.mode_significance()
    print("Top 5 significant modes:")
    print(mode_info.head(5))
    
    # 4. Plot DMD spectrum
    print("\nPlotting DMD spectrum...")
    fig = tdmd.plot_spectrum()
    fig.savefig('results/dmd_spectrum.png', dpi=300, bbox_inches='tight')
    
    # 5. Visualize the most significant modes
    print("\nVisualizing top DMD modes...")
    fig = tdmd.visualize_modes(n_modes=3)
    fig.savefig('results/dmd_modes.png', dpi=300, bbox_inches='tight')
    
    # 6. Reconstruct and evaluate model accuracy
    print("\nReconstructing data and evaluating accuracy...")
    reconstruction = tdmd.reconstruct()
    error_analysis = tdmd.residual_analysis()
    print(f"Relative error: {error_analysis['relative_error']:.6f}")
    print(f"Max error: {error_analysis['max_error']:.6f}")
    print(f"Mean error: {error_analysis['mean_error']:.6f}")
    
    # 7. Plot reconstruction comparison
    print("\nPlotting reconstruction comparison...")
    fig = tdmd.plot_reconstruction_comparison(n_snapshots=3)
    fig.savefig('results/reconstruction_comparison.png', dpi=300, bbox_inches='tight')
    
    # 8. Plot reconstruction error over time
    print("\nPlotting reconstruction error over time...")
    fig = tdmd.plot_reconstruction_error()
    fig.savefig('results/reconstruction_error.png', dpi=300, bbox_inches='tight')
    
    # 9. Make future predictions
    print("\nMaking future predictions...")
    future_times = np.linspace(0, n_time * 0.1 * 1.5, 120)  # Extend 50% beyond training data
    predictions, confidence = tdmd.forecast(future_times)
    
    # 10. Generate diagnostic report
    print("\nGenerating diagnostic report...")
    report = tdmd.generate_diagnostic_report()
    with open('results/diagnostic_report.md', 'w') as f:
        f.write(report)
    
    # 11. Plot mode contributions
    print("\nPlotting mode contributions...")
    fig = tdmd.plot_mode_contributions(n_modes=5)
    fig.savefig('results/mode_contributions.png', dpi=300, bbox_inches='tight')
    
    # 12. Calculate and plot SVD analysis
    print("\nPerforming SVD analysis...")
    fig = tdmd.plot_svd_analysis()
    fig.savefig('results/svd_analysis.png', dpi=300, bbox_inches='tight')
    
    print("\nTest completed successfully. Results saved to 'results/' directory.")


if __name__ == "__main__":
    main()
