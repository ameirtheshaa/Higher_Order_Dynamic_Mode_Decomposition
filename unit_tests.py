#!/usr/bin/env python3
"""
Unit tests for the TensorDMD class.

Run with:
    pytest test_tensor_dmd_unittest.py -v
"""

import unittest
import numpy as np
import pytest
from tensor_dmd import TensorDMD


class TestTensorDMD(unittest.TestCase):
    """Test cases for the TensorDMD class."""

    def setUp(self):
        """Set up test data before each test method."""
        # Generate a small synthetic dataset for testing
        self.n_time, self.nx, self.ny = 20, 10, 10
        self.test_data = TensorDMD.generate_example_tensor(
            n_time=self.n_time,
            nx=self.nx,
            ny=self.ny,
            freq1=0.1,
            freq2=0.05,
            noise_level=0.01
        )
        
        # Create a TensorDMD instance
        self.tdmd = TensorDMD(
            self.test_data,
            dt=0.1,
            rank=(None, 5, 5),
            truncate_rank=5
        )

    def test_initialization(self):
        """Test initialization of TensorDMD instance."""
        self.assertEqual(self.tdmd.dims, (self.n_time, self.nx, self.ny))
        self.assertEqual(self.tdmd.n_time_steps, self.n_time)
        self.assertEqual(self.tdmd.dt, 0.1)
        self.assertEqual(self.tdmd.rank, (None, 5, 5))
        self.assertEqual(self.tdmd.truncate_rank, 5)
        
        # Test X1 and X2 shapes
        self.assertEqual(self.tdmd.X1.shape, (self.n_time-1, self.nx, self.ny))
        self.assertEqual(self.tdmd.X2.shape, (self.n_time-1, self.nx, self.ny))

    def test_core_and_factors(self):
        """Test core tensor and factors from Tucker decomposition."""
        # Check that core tensor and factors exist
        self.assertIsNotNone(self.tdmd.core)
        self.assertIsNotNone(self.tdmd.factors)
        
        # Check the number of factors
        self.assertEqual(len(self.tdmd.factors), 3)  # 3D tensor: time, x, y

    def test_dmd_computation(self):
        """Test DMD computation results."""
        # Check eigenvalues and eigenvectors
        self.assertEqual(len(self.tdmd.Lambda), 5)  # Due to truncate_rank=5
        self.assertEqual(self.tdmd.W.shape[1], 5)  # Number of eigenvectors
        
        # Check omega (frequencies) shape
        self.assertEqual(len(self.tdmd.omega), 5)
        
        # Check alpha (amplitudes) shape
        self.assertEqual(len(self.tdmd.alpha), 5)

    def test_reconstruction(self):
        """Test data reconstruction."""
        reconstructed = self.tdmd.reconstruct()
        
        # Check shape
        self.assertEqual(reconstructed.shape, self.test_data.shape)
        
        # Check reconstruction error is reasonable
        residual_info = self.tdmd.residual_analysis()
        self.assertLess(residual_info['relative_error'], 0.5)  # Arbitrary threshold

    def test_prediction(self):
        """Test future state prediction."""
        future_times = np.linspace(0, 3.0, 31)
        predictions = self.tdmd.predict(future_times)
        
        # Check shape
        self.assertEqual(predictions.shape, (31, self.nx, self.ny))
        
        # Test forecast function with confidence
        forecasted, confidence = self.tdmd.forecast(future_times)
        self.assertEqual(forecasted.shape, (31, self.nx, self.ny))
        self.assertEqual(len(confidence), 31)

    def test_mode_analysis(self):
        """Test mode analysis functions."""
        # Test mode frequencies
        freq_df = self.tdmd.mode_frequencies()
        self.assertEqual(len(freq_df), 5)
        
        # Test mode significance
        sig_df = self.tdmd.mode_significance()
        self.assertEqual(len(sig_df), 5)
        
        # Test compute_mode_energy
        energies = self.tdmd.compute_mode_energy()
        self.assertEqual(len(energies), 5)
        self.assertAlmostEqual(np.sum(energies), 1.0, places=6)

    def test_visualization_methods(self):
        """Test that visualization methods return figure objects."""
        # Test spectrum plot
        fig = self.tdmd.plot_spectrum()
        self.assertIsNotNone(fig)
        
        # Test mode amplitude plot
        fig = self.tdmd.plot_mode_amplitudes()
        self.assertIsNotNone(fig)
        
        # Test mode frequency plot
        fig = self.tdmd.plot_mode_frequencies()
        self.assertIsNotNone(fig)
        
        # Test growth rate plot
        fig = self.tdmd.plot_growth_rates()
        self.assertIsNotNone(fig)
        
        # Test reconstruction error plot
        fig = self.tdmd.plot_reconstruction_error()
        self.assertIsNotNone(fig)
        
        # Test mode visualization
        fig = self.tdmd.visualize_modes(n_modes=2)
        self.assertIsNotNone(fig)
        
        # Test reconstruction comparison
        fig = self.tdmd.plot_reconstruction_comparison(n_snapshots=2)
        self.assertIsNotNone(fig)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test initialization with 1D data (should raise ValueError)
        with self.assertRaises(ValueError):
            TensorDMD(np.zeros(10), dt=0.1)
            
        # Test with None data (should raise TypeError)
        with self.assertRaises(TypeError):
            TensorDMD(None, dt=0.1)

    def test_is_mode_physical(self):
        """Test mode physicality detection."""
        # Test at least one mode
        result = self.tdmd.is_mode_physical(0, threshold=0.01)
        self.assertIsInstance(result, bool)

    def test_generate_diagnostic_report(self):
        """Test diagnostic report generation."""
        report = self.tdmd.generate_diagnostic_report()
        self.assertIsInstance(report, str)
        self.assertIn("Tensor DMD Analysis Diagnostic Report", report)
        self.assertIn("System Stability", report)
        self.assertIn("Reconstruction Accuracy", report)


if __name__ == '__main__':
    unittest.main()
