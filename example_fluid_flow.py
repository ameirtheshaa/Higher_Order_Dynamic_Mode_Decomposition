"""
Example: Fluid Flow Analysis using TensorDMD

This example demonstrates using TensorDMD to analyze a simulated fluid flow dataset.
The data consists of a 3D tensor representing a flow field evolving over time.

"""

import numpy as np
import matplotlib.pyplot as plt
from tensor_dmd import TensorDMD
import os


def generate_fluid_flow_data(n_time=100, nx=50, ny=50, n_vortices=3, diffusion=0.01):
    """
    Generate synthetic fluid flow data with vortices and diffusion.
    
    Parameters
    ----------
    n_time : int
        Number of time steps
    nx, ny : int
        Spatial dimensions
    n_vortices : int
        Number of vortices in the flow
    diffusion : float
        Diffusion rate
        
    Returns
    -------
    numpy.ndarray
        Flow field tensor of shape (n_time, nx, ny)
    """
    # Initialize grid
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize flow field
    flow = np.zeros((n_time, nx, ny))
    
    # Generate initial vortices
    vortex_centers = []
    vortex_strengths = []
    for i in range(n_vortices):
        # Random positions for vortices
        x_pos = np.random.uniform(-0.7, 0.7)
        y_pos = np.random.uniform(-0.7, 0.7)
        vortex_centers.append((x_pos, y_pos))
        
        # Random strengths (positive = counterclockwise, negative = clockwise)
        strength = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
        vortex_strengths.append(strength)
    
    # Create initial flow field with vortices
    flow_t = np.zeros((nx, ny))
    for (x_pos, y_pos), strength in zip(vortex_centers, vortex_strengths):
        # Create vortex using Gaussian field
        dist = np.sqrt((X - x_pos)**2 + (Y - y_pos)**2)
        vortex = strength * np.exp(-dist**2 / 0.1)
        flow_t += vortex
    
    flow[0] = flow_t
    
    # Simulate time evolution with diffusion and advection
    for t in range(1, n_time):
        # Apply simple diffusion
        flow_t = flow[t-1] + diffusion * np.random.randn(nx, ny)
        
        # Add time-varying perturbation
        if t % 10 == 0:
            # Add a new small vortex
            x_pos = np.random.uniform(-0.7, 0.7)
            y_pos = np.random.uniform(-0.7, 0.7)
            strength = np.random.choice([-0.5, 0.5])
            
            dist = np.sqrt((X - x_pos)**2 + (Y - y_pos)**2)
            new_vortex = strength * np.exp(-dist**2 / 0.05)
            flow_t += new_vortex
        
        # Store the result
        flow[t] = flow_t
    
    return flow


def main():
    """Run the fluid flow analysis example."""
    print("Fluid Flow Analysis Example using TensorDMD")
    
    # Create output directory
    os.makedirs('results/fluid_flow', exist_ok=True)
    
    # 1. Generate synthetic fluid flow data
    print("\nGenerating synthetic fluid flow data...")
    flow_data = generate_fluid_flow_data(
        n_time=80, 
        nx=50, 
        ny=50, 
        n_vortices=4,
        diffusion=0.01
    )
    print(f"Flow data shape: {flow_data.shape}")
    
    # 2. Visualize the original flow data at selected time points
    print("\nVisualizing original flow data...")
    time_points = [0, 20, 40, 60, 79]
    fig, axes = plt.subplots(1, len(time_points), figsize=(15, 3))
    
    for i, t in enumerate(time_points):
        im = axes[i].imshow(flow_data[t], cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        axes[i].set_title(f"t = {t}")
        axes[i].axis('off')
    
    plt.colorbar(im, ax=axes, shrink=0.8, label='Vorticity')
    plt.tight_layout()
    plt.savefig('results/fluid_flow/original_flow.png', dpi=300, bbox_inches='tight')
    
    # 3. Create and run TensorDMD analysis
    print("\nRunning TensorDMD analysis...")
    tdmd = TensorDMD(
        flow_data,
        dt=0.1,
        rank=(None, 20, 20),  # Reduce spatial dimensions
        truncate_rank=10  # Keep top 10 modes
    )
    
    # 4. Analyze DMD modes
    print("\nAnalyzing DMD modes...")
    mode_info = tdmd.mode_significance()
    print("Top 5 significant modes:")
    print(mode_info.head(5))
    
    # 5. Visualize DMD modes
    print("\nVisualizing DMD modes...")
    fig = tdmd.visualize_modes(n_modes=4)
    plt.savefig('results/fluid_flow/dmd_modes.png', dpi=300, bbox_inches='tight')
    
    # 6. Plot DMD spectrum
    print("\nPlotting DMD spectrum...")
    fig = tdmd.plot_spectrum()
    plt.savefig('results/fluid_flow/dmd_spectrum.png', dpi=300, bbox_inches='tight')
    
    # 7. Reconstruct the flow
    print("\nReconstructing flow data...")
    reconstruction = tdmd.reconstruct()
    
    # 8. Evaluate reconstruction accuracy
    error_info = tdmd.residual_analysis()
    print(f"Relative reconstruction error: {error_info['relative_error']:.6f}")
    
    # 9. Compare original and reconstructed flow
    print("\nComparing original and reconstructed flow...")
    fig = tdmd.plot_reconstruction_comparison(time_indices=time_points[:3])
    plt.savefig('results/fluid_flow/flow_reconstruction.png', dpi=300, bbox_inches='tight')
    
    # 10. Future state prediction
    print("\nPredicting future flow states...")
    future_times = np.linspace(0, 12, 120)  # Predict to 50% beyond training data
    predictions, confidence = tdmd.forecast(future_times)
    
    # 11. Visualize predictions
    print("\nVisualizing flow predictions...")
    future_points = [40, 80, 100, 119]  # Mix of training and prediction range
    
    fig, axes = plt.subplots(1, len(future_points), figsize=(15, 3))
    
    for i, t in enumerate(future_points):
        im = axes[i].imshow(predictions[t], cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        time_val = future_times[t]
        conf = confidence.get(time_val, 0)
        
        title = f"t = {time_val:.1f}"
        if t >= 80:  # Beyond training data
            title += f"\nconf: {conf:.2f}"
            
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.colorbar(im, ax=axes, shrink=0.8, label='Vorticity')
    plt.tight_layout()
    plt.savefig('results/fluid_flow/flow_predictions.png', dpi=300, bbox_inches='tight')
    
    # 12. Generate comprehensive diagnostic report
    print("\nGenerating diagnostic report...")
    report = tdmd.generate_diagnostic_report()
    with open('results/fluid_flow/flow_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("\nFluid flow analysis completed. Results saved to 'results/fluid_flow/' directory.")


if __name__ == "__main__":
    main()
