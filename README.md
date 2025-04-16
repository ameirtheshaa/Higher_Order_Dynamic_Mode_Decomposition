# TensorDMD

A Python library for Tensor-based Dynamic Mode Decomposition (DMD) analysis of dynamical systems.

## Overview

TensorDMD extends the traditional Dynamic Mode Decomposition (DMD) to handle multi-dimensional tensor data. It leverages Tucker decomposition (Higher-Order Singular Value Decomposition) to efficiently analyze complex spatiotemporal dynamics in high-dimensional systems.

Key features:
- Tensor DMD computation using HOSVD (Tucker decomposition)
- Mode analysis and diagnostics
- Future state prediction
- Comprehensive visualization methods
- Robust error analysis and validation tools

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tensor-dmd.git
cd tensor-dmd

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- numpy
- tensorly
- matplotlib
- pandas
- seaborn
- IPython

## Quick Start

```python
import numpy as np
from tensor_dmd import TensorDMD

# Generate example data
n_time, nx, ny = 50, 20, 20
tensor_data = TensorDMD.generate_example_tensor(n_time, nx, ny)

# Create TensorDMD instance
tdmd = TensorDMD(tensor_data, dt=0.1, rank=(None, 10, 10))

# Analyze modes
mode_info = tdmd.mode_significance()
print(mode_info.head())

# Plot spectrum
fig = tdmd.plot_spectrum()
fig.savefig('dmd_spectrum.png')

# Reconstruct and compute error
reconstruction = tdmd.reconstruct()
error_analysis = tdmd.residual_analysis()
print(f"Relative error: {error_analysis['relative_error']:.4e}")

# Make predictions
future_times = np.linspace(0, 8, 80)
predictions, confidence = tdmd.forecast(future_times)
```

## Example

See the `examples/` directory for detailed examples and tutorials:

- `basic_example.py`: Simple example with synthetic data
- `fluid_flow_analysis.py`: Analysis of fluid flow data
- `parameter_tuning.py`: How to tune TensorDMD parameters

## Documentation

The main class `TensorDMD` provides the following key methods:

- `__init__(data, dt=1.0, rank=None, truncate_rank=None)`: Initialize with tensor data
- `reconstruct(times=None)`: Reconstruct the data at specified times
- `predict(times)`: Predict future states of the system
- `forecast(future_times)`: Make forecasts with confidence estimates
- `residual_analysis()`: Assess the quality of the approximation

Visualization methods:
- `plot_spectrum()`: Plot the DMD eigenvalues in the complex plane
- `plot_mode_amplitudes()`: Plot the amplitudes of the DMD modes
- `visualize_modes()`: Visualize the spatial structure of selected modes
- `plot_reconstruction_comparison()`: Compare original and reconstructed snapshots
- `plot_reconstruction_error()`: Plot the reconstruction error over time
- `create_mode_animation()`: Create an animation of a DMD mode over time

Analysis methods:
- `mode_frequencies()`: Calculate frequency and growth rate of each mode
- `mode_significance()`: Calculate the significance of each mode
- `compute_optimal_rank()`: Find optimal Tucker ranks via cross-validation
- `generate_diagnostic_report()`: Generate a comprehensive diagnostic report

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{tensor-dmd,
  author = {Ameir Shaa, Claude Guet},
  title = {},
  year = {2025},
  publisher = {},
  journal = {},
  howpublished = {}
}
```
