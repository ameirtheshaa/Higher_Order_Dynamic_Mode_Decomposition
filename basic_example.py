"""
Basic Example: Using TensorDMD for synthetic data analysis

This example demonstrates the basic usage of TensorDMD with a simple
synthetic dataset containing oscillatory patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensor_dmd import TensorDMD
import os


def main():
    """Run a basic example of TensorDMD."""
    print("Basic Example: TensorDMD")
    
    # Create output directory
    os.makedirs('results/basic_example', exist_ok=True)
    
    # 1. Generate synthetic data with two oscillatory patterns
    print("\nGenerating synthetic data...")
    n_time, nx, ny = 50, 20, 20
    tensor_data = TensorDMD.generate_example_tensor(
        n_time=n_time,
        nx=nx, 
        ny=ny,
        freq1=0.1,  # First frequency component
        freq2=0.05,  # Second frequency component
        noise_level=0.02  # Add some noise
    )
    print(f"Data shape: {tensor_data.shape}")
    
    # 2. Visualize the original data
    print("\nVisualizing original data...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Show data at beginning, middle, and end
    for i, t in enumerate([0, n_time//2, n_time-1]):
        im = axes[i].imshow(tensor_data[t], cmap='viridis')
        axes[i].set_title(f"t = {t}")
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('results/basic_example/original_data.png', dpi=300)
    
    # 3. Create a TensorDMD instance and perform analysis
    print("\nPerforming TensorDMD analysis...")
    tdmd = TensorDMD(
        tensor_data,
        dt=0.1,
        rank=(None, 10, 10),  # Reduce spatial dimensions to rank 10
        truncate_rank=8  # Keep only 8 DMD modes
    )
    
    # 4. Show DMD spectrum
    print("\nPlotting DMD spectrum...")
    fig = tdmd.plot_spectrum()
    plt.savefig('results/basic_example/dmd_spectrum.png', dpi=300)
    
    # 5. Show significant modes
    print("\nListing significant modes...")
    mode_info = tdmd.mode_significance()
    print(mode_info.head())
    
    # 6. Visualize DMD modes
    print("\nVisualizing DMD modes...")
    fig = tdmd.visualize_modes(n_modes=3)
    plt.savefig('results/basic_example/dmd_modes.png', dpi=300)
    
    # 7. Reconstruct data and compute error
    print("\nReconstructing data...")
    reconstructed = tdmd.reconstruct()
    
    error_info = tdmd.residual_analysis()
    print(f"Reconstruction relative error: {error_info['relative_error']:.6f}")
    
    # 8. Compare original and reconstructed data
    print("\nComparing original vs reconstructed data...")
    fig = tdmd.plot_reconstruction_comparison(n_snapshots=3)
    plt.savefig('results/basic_example/reconstruction_comparison.png', dpi=300)
    
    # 9. Show mode frequencies and amplitudes
    print("\nAnalyzing mode frequencies and amplitudes...")
    fig = tdmd.plot_mode_frequencies()
    plt.savefig('results/basic_example/mode_frequencies.png', dpi=300)
    
    fig = tdmd.plot_mode_amplitudes()
    plt.savefig('results/basic_example/mode_amplitudes.png', dpi=300)
    
    # 10. Make future predictions
    print("\nMaking future predictions...")
    future_times = np.linspace(0, n_time * 0.1 * 1.5, 75)  # Predict 50% beyond training data
    predictions, confidence = tdmd.forecast(future_times)
    
    # 11. Show predictions at selected time points
    print("\nVisualizing predictions...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Times within training data
    for i, t in enumerate([10, 25, 40]):
        t_idx = int(t / 0.1)
        im = axes[0, i].imshow(predictions[t_idx], cmap='viridis')
        axes[0, i].set_title(f"Prediction at t = {t:.1f}\nconf: {confidence[t]:.2f}")
        fig.colorbar(im, ax=axes[0, i])
    
    # Times beyond training data
    for i, t in enumerate([5.5, 6.5, 7.5]):
        t_idx = int(t / 0.1)
        im = axes[1, i].imshow(predictions[t_idx], cmap='viridis')
        axes[1, i].set_title(f"Forecast at t = {t:.1f}\nconf: {confidence[t]:.2f}")
        fig.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig('results/basic_example/predictions.png', dpi=300)
    
    print("\nExample completed. Results saved to 'results/basic_example/'")


if __name__ == "__main__":
    main()
