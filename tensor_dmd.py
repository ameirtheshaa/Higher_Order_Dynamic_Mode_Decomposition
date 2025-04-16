import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from IPython.display import display
import matplotlib.animation as animation


class TensorDMD:
    """
    Tensor-based Dynamic Mode Decomposition (DMD) class for analyzing dynamical systems.

    This implementation provides a comprehensive set of tools for tensor DMD analysis including:
    - Tensor DMD computation using HOSVD (Tucker decomposition)
    - Mode analysis and diagnostics
    - Future state prediction
    - Various visualization methods

    Parameters
    ----------
    data : numpy.ndarray
        The input tensor of shape (time_steps, spatial_dim1, spatial_dim2, ...)
        First dimension is assumed to be time.
    dt : float, optional
        Time step between snapshots, default is 1.0.
    rank : tuple or None, optional
        Rank of the Tucker decomposition for each mode. If None, the original
        dimensions are preserved.
    truncate_rank : int or None, optional
        Number of DMD modes to retain. If None, all modes are kept.
    """

    def __init__(self, data, dt=1.0, rank=None, truncate_rank=None):
        """
        Initialize the TensorDMD object with data and optional parameters.

        Parameters
        ----------
        data : numpy.ndarray
            The input tensor of shape (time_steps, spatial_dim1, spatial_dim2, ...)
            First dimension is assumed to be time.
        dt : float, optional
            Time step between snapshots, default is 1.0.
        rank : tuple or None, optional
            Rank of the Tucker decomposition for each mode. If None, the original
            dimensions are preserved.
        truncate_rank : int or None, optional
            Number of DMD modes to retain. If None, all modes are kept.
        """
        # Set tensorly backend
        tl.set_backend('numpy')

        if len(data.shape) < 2:
            raise ValueError("Input data must be at least a 2D tensor (matrix)")

        self.data = data
        self.dt = dt
        self.rank = rank
        self.truncate_rank = truncate_rank

        # Get dimensions
        self.dims = data.shape
        self.n_time_steps = self.dims[0]

        # Split tensor into X1 and X2 (offset by one time step)
        self.X1 = data[:-1]  # All time steps except the last one
        self.X2 = data[1:]   # All time steps except the first one

        # Compute tensor DMD
        self._compute_tensor_dmd()

    def _compute_tensor_dmd(self):
        """Compute the Tensor-based Dynamic Mode Decomposition with extensive debugging output."""

        # --- 1. Tucker Decomposition on X1 ---
        # Decompose the tensor using Tucker decomposition with the provided rank.
        self.core, self.factors = tucker(self.X1, rank=self.rank, init='svd')

        # Debug: Print the shapes of the core and the factor matrices.
        print("Tucker decomposition:")
        print("Core shape:", self.core.shape)
        print("Factor shapes:", [f.shape for f in self.factors])

        # Get the time factor (assumed to be the first factor corresponding to the temporal mode).
        self.time_factors = self.factors[0]

        # --- 2. Projection onto the Reduced Space ---
        # Project both X1 and X2 onto the reduced space using the transposed factor matrices.
        self.X1_reduced = tl.tenalg.multi_mode_dot(
            self.X1, [f.T for f in self.factors], modes=range(len(self.factors))
        )
        self.X2_reduced = tl.tenalg.multi_mode_dot(
            self.X2, [f.T for f in self.factors], modes=range(len(self.factors))
        )

        # Debug: Print shapes of reduced tensors.
        print("Reduced tensor shapes:")
        print("X1_reduced shape:", self.X1_reduced.shape)
        print("X2_reduced shape:", self.X2_reduced.shape)

        # --- 3. Unfold the Reduced Tensors ---
        # Unfold along the first mode (time) so that each column is a snapshot.
        self.X1_mat = tl.unfold(self.X1_reduced, mode=0).T
        self.X2_mat = tl.unfold(self.X2_reduced, mode=0).T
        print("Matricized data shapes:")
        print("X1_mat shape:", self.X1_mat.shape)
        print("X2_mat shape:", self.X2_mat.shape)

        # --- 4. Compute DMD Operator via Eigen-decomposition ---
        # Compute the pseudoinverse of the unfolded X1 and then the reduced DMD operator.
        self.X1_pinv = np.linalg.pinv(self.X1_mat)
        self.A_tilde = self.X2_mat @ self.X1_pinv
        eigenvalues, eigenvectors = np.linalg.eig(self.A_tilde)

        # Sort eigenvalues (and corresponding eigenvectors) by their magnitude in descending order.
        sort_idx = np.argsort(np.abs(eigenvalues))[::-1]
        self.Lambda = eigenvalues[sort_idx]
        self.W = eigenvectors[:, sort_idx]

        # Optional truncation if truncate_rank is set.
        if self.truncate_rank is not None:
            self.Lambda = self.Lambda[:self.truncate_rank]
            self.W = self.W[:, :self.truncate_rank]

        # Reduced DMD modes.
        self.phi_reduced = self.W
        print("DMD eigen-decomposition:")
        print("Lambda shape:", self.Lambda.shape)
        print("W shape (phi_reduced):", self.phi_reduced.shape)

        # --- 5. Reconstruct DMD Modes in the Original Space ---
        # The spatial dimensions come from the Tucker core (all modes except the time mode).
        spatial_shape = self.core.shape[1:]
        prod_spatial = np.prod(spatial_shape)
        print("Spatial Tucker ranks:", spatial_shape, "Product:", prod_spatial)
        print("phi_reduced first dimension:", self.phi_reduced.shape[0])

        if self.phi_reduced.shape[0] != prod_spatial:
            print("Warning: The product of spatial Tucker ranks does not match the first dimension of phi_reduced.")

        # Reshape phi_reduced into a tensor with shape (r1, r2, ..., n_modes)
        # where r1, r2, ... are the spatial Tucker ranks.
        modes_tensor = tl.tensor(self.phi_reduced.reshape(spatial_shape + (-1,)))
        print("Modes tensor shape after reshaping:", modes_tensor.shape)

        # Multiply the reshaped modes tensor with the spatial factor matrices (skipping the time factor).
        # Note: The spatial axes of modes_tensor are now 0, 1, ...; hence we use modes=range(len(self.factors)-1)
        self.phi = tl.tenalg.multi_mode_dot(
            modes_tensor, self.factors[1:], modes=range(len(self.factors) - 1)
        )
        print("Reconstructed DMD modes (phi) shape:", self.phi.shape)

        # --- 6. Compute Continuous-time DMD Frequencies ---
        # The frequencies are computed from the eigenvalues.
        self.omega = np.log(self.Lambda) / self.dt
        print("Computed DMD frequencies (omega):", self.omega)

        # --- 7. Compute Mode Amplitudes via the Initial Condition ---
        # Get initial condition from the data.
        x0 = self.data[0]

        # Project the initial condition using the spatial factors. The projection is done along spatial modes only.
        x0_reduced = tl.tenalg.multi_mode_dot(
            x0, [f.T for f in self.factors[1:]], modes=range(len(self.factors) - 1)
        )
        x0_vec = x0_reduced.flatten()

        # Solve a least-squares problem to get the amplitudes.
        modes_mat = self.phi_reduced.reshape(self.phi_reduced.shape[0], -1)
        self.alpha = np.linalg.lstsq(modes_mat, x0_vec, rcond=None)[0]
        print("Computed DMD mode amplitudes (alpha):", self.alpha)

        # --- 8. Store Additional Attributes ---
        self.dmd_spectrum = np.abs(self.Lambda)
        self.amplitudes = np.abs(self.alpha)
        self.eigenvalues = self.Lambda

    def reconstruct(self, times=None):
        """
        Reconstruct the data using the tensor DMD modes.

        Parameters
        ----------
        times : array-like, optional
            Times at which to reconstruct the data. If None, uses the original snapshot times.

        Returns
        -------
        numpy.ndarray
            Reconstructed tensor at specified times.
        """
        if times is None:
            times = np.arange(self.n_time_steps) * self.dt

        n_times = len(times)
        reconstructed = np.zeros((n_times,) + self.dims[1:], dtype=complex)

        for i, t in enumerate(times):
            # Compute time dynamics
            time_dynamics = np.diag(np.power(self.Lambda, t/self.dt))
            # Compute reduced state
            temp = np.dot(self.phi_reduced, np.dot(time_dynamics, self.alpha))
            # Reshape to tensor format
            temp_tensor = temp.reshape(self.X1_reduced.shape[1:])
            # Transform back to original space
            reconstructed[i] = tl.tenalg.multi_mode_dot(
                temp_tensor, self.factors[1:], modes=range(len(self.factors[1:]))
            )

        # For real-valued data, take the real part
        if np.isreal(self.data).all():
            reconstructed = reconstructed.real

        return reconstructed

    def predict(self, times):
        """
        Predict future states of the system.

        Parameters
        ----------
        times : array-like
            Times at which to predict the state of the system.

        Returns
        -------
        numpy.ndarray
            Predicted states at specified times.
        """
        return self.reconstruct(times)

    def mode_frequencies(self):
        """
        Calculate the frequency and growth rate of each mode.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing mode index, frequency, growth rate,
            magnitude, and normalized amplitude.
        """
        frequencies = np.imag(self.omega) / (2 * np.pi)
        growth_rates = np.real(self.omega)
        magnitudes = np.abs(self.Lambda)
        amplitudes = np.abs(self.alpha)
        norm_amplitudes = amplitudes / np.sum(amplitudes)

        data = {
            'Mode': np.arange(len(self.Lambda)),
            'Frequency': frequencies,
            'Growth_Rate': growth_rates,
            'Magnitude': magnitudes,
            'Amplitude': amplitudes,
            'Normalized_Amplitude': norm_amplitudes
        }

        return pd.DataFrame(data).sort_values(by='Normalized_Amplitude', ascending=False)

    def mode_significance(self):
        """
        Calculate the significance of each mode based on its amplitude and growth/decay.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing modes sorted by their significance.
        """
        # Calculate significance as a combination of amplitude and magnitude
        amplitudes = np.abs(self.alpha)
        magnitudes = np.abs(self.Lambda)

        # Normalize amplitudes to [0,1]
        norm_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes

        # Calculate significance score
        significance = norm_amplitudes * magnitudes

        data = {
            'Mode': np.arange(len(self.Lambda)),
            'Significance': significance,
            'Normalized_Amplitude': norm_amplitudes,
            'Magnitude': magnitudes
        }

        return pd.DataFrame(data).sort_values(by='Significance', ascending=False)

    def residual_analysis(self):
        """
        Perform residual analysis to assess the quality of the tensor DMD approximation.

        Returns
        -------
        dict
            Dictionary containing residual statistics.
        """
        # Reconstruct the data
        reconstructed = self.reconstruct()

        # Calculate residuals
        residuals = self.data - reconstructed

        # Calculate error metrics
        relative_error = np.linalg.norm(residuals.reshape(-1)) / np.linalg.norm(self.data.reshape(-1))
        max_error = np.max(np.abs(residuals))
        mean_error = np.mean(np.abs(residuals))

        return {
            'relative_error': relative_error,
            'max_error': max_error,
            'mean_error': mean_error,
            'residuals': residuals
        }

    def eigenvalue_check(self):
        """
        Check the validity of DMD eigenvalues.

        Returns
        -------
        float
            Spectral radius of DMD operator A.
        """
        # Check if any eigenvalues have magnitude > 1 (indicating growth)
        growing_modes = np.sum(np.abs(self.Lambda) > 1.0)
        print(f"Number of growing modes (|λ| > 1): {growing_modes}")

        # Check spectral radius
        spectral_radius = np.max(np.abs(self.Lambda))
        print(f"Spectral radius: {spectral_radius}")

        return spectral_radius

    def plot_spectrum(self, figsize=(10, 8), size_factor=100):
        """
        Plot the DMD spectrum in the complex plane.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        size_factor : float, optional
            Scaling factor for the size of the points.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the spectrum plot.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.6)

        # Plot eigenvalues
        amplitudes = np.abs(self.alpha)
        normalized_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes

        scatter = ax.scatter(self.Lambda.real, self.Lambda.imag,
                             s=normalized_amplitudes*size_factor,
                             c=np.abs(self.Lambda),
                             cmap='viridis',
                             alpha=0.7)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Magnitude |λ|')

        # Set labels and title
        ax.set_xlabel('Real(λ)')
        ax.set_ylabel('Imag(λ)')
        ax.set_title('DMD Spectrum (Discrete Eigenvalues)')

        # Add grid and equal aspect ratio
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Annotate significant modes
        significant_modes = np.argsort(normalized_amplitudes)[-5:]
        for i in significant_modes:
            ax.annotate(f'{i}',
                        (self.Lambda[i].real, self.Lambda[i].imag),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=10)

        return fig
        
    def plot_complex_alpha(self, figsize=(8, 8), annotate=False):
        """
        Plot the complex mode amplitudes in the complex plane.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        annotate : bool, optional
            Whether to annotate the points with mode indices.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the complex plane plot.
        """
        z = self.alpha

        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        scatter = ax.scatter(z.real, z.imag, c=np.abs(z), cmap='viridis', alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Amplitude |α|')

        # Annotate points if requested
        if annotate:
            for i, val in enumerate(z):
                ax.text(z[i].real + 0.1, z[i].imag, f'{i}', fontsize=9)

        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title('Mode Amplitudes in Complex Plane')
        ax.grid(True)
        ax.set_aspect('equal')

        return fig

    def forecast(self, future_times):
        """
        Forecast future states of the system.

        Parameters
        ----------
        future_times : array-like
            Times at which to forecast the state of the system.

        Returns
        -------
        numpy.ndarray
            Forecasted states at specified times.
        dict
            Dictionary containing confidence information.
        """
        # Forecast using DMD
        forecasted_states = self.predict(future_times)

        # Calculate confidence based on extrapolation distance and system stability
        spectral_radius = np.max(np.abs(self.Lambda))
        max_observed_time = (self.n_time_steps - 1) * self.dt
        confidence = {}

        for i, t in enumerate(future_times):
            # For times within observed range, confidence is high
            if t <= max_observed_time:
                confidence_value = 0.9
            else:
                # Decrease confidence based on extrapolation distance and stability
                extrapolation_factor = (t - max_observed_time) / max_observed_time

                # For stable systems, confidence decreases more slowly
                if spectral_radius <= 1.0:
                    confidence_value = 0.9 * np.exp(-0.5 * extrapolation_factor)
                else:
                    confidence_value = 0.9 * np.exp(-2.0 * extrapolation_factor * spectral_radius)

                confidence_value = max(0.1, confidence_value)  # Set minimum confidence

            confidence[t] = confidence_value

        return forecasted_states, confidence
        
    def generate_diagnostic_report(self):
        """
        Generate a comprehensive diagnostic report of the tensor DMD analysis.

        Returns
        -------
        str
            Report text containing diagnostic information.
        """
        # Calculate basic metrics
        residual_info = self.residual_analysis()
        spectral_radius = np.max(np.abs(self.Lambda))

        # Create mode frequency table
        mode_freq_df = self.mode_frequencies().head(10)

        # Calculate system stability
        is_stable = spectral_radius <= 1.0

        # Identify dominant modes
        significant_modes = self.mode_significance().head(5)

        # Generate report text
        report = "# Tensor DMD Analysis Diagnostic Report\n\n"

        # Basic information
        report += "## System Overview\n"
        report += f"- Tensor dimensions: {self.dims}\n"
        report += f"- Number of time steps: {self.n_time_steps}\n"
        report += f"- Time step (dt): {self.dt}\n"
        report += f"- Tucker decomposition rank: {self.rank}\n"
        report += f"- DMD truncation rank: {self.truncate_rank}\n\n"

        # System stability
        report += "## System Stability\n"
        report += f"- Spectral radius: {spectral_radius:.4f}\n"
        report += f"- System stability: {'Stable' if is_stable else 'Unstable'}\n"
        if not is_stable:
            report += "  ⚠️ The system has growing modes which indicate instability or transient growth.\n"

        # Reconstruction accuracy
        report += "\n## Reconstruction Accuracy\n"
        report += f"- Relative error: {residual_info['relative_error']:.4e}\n"
        report += f"- Maximum absolute error: {residual_info['max_error']:.4e}\n"
        report += f"- Mean absolute error: {residual_info['mean_error']:.4e}\n"

        # Most significant modes
        report += "\n## Most Significant Modes\n"
        report += significant_modes.to_markdown() + "\n"

        # Mode frequencies
        report += "\n## Mode Frequency Analysis\n"
        report += mode_freq_df.to_markdown() + "\n"

        # Recommendations
        report += "\n## Recommendations\n"

        # Rank selection recommendation
        if self.rank is None:
            report += "- Consider specifying Tucker decomposition rank to reduce dimensionality.\n"

        # Stability recommendations
        if not is_stable:
            report += "- Investigate growing modes to understand system instabilities.\n"
            report += "- For prediction tasks, be cautious about extrapolating too far into the future.\n"

        # Mode selection recommendations
        report += f"- Focus analysis on the top {min(5, len(significant_modes))} modes by significance for key dynamics.\n"

        # Model interpretation
        report += "\n## Physical Interpretation\n"
        report += "- DMD modes represent coherent structures in the tensor data with specific frequencies and growth/decay rates.\n"
        report += "- Modes with eigenvalues close to the unit circle represent persistent dynamics.\n"
        report += "- Modes with eigenvalues inside the unit circle represent decaying dynamics.\n"
        report += "- Modes with eigenvalues outside the unit circle represent growing dynamics.\n"

        return report
        
    def create_mode_animation(self, mode_idx, t_span=None, n_frames=50, figsize=(10, 8)):
        """
        Create an animation of a DMD mode over time.

        Parameters
        ----------
        mode_idx : int
            Index of the mode to animate.
        t_span : tuple, optional
            Time span for animation (t_start, t_end). If None, uses the original time span.
        n_frames : int, optional
            Number of frames in the animation.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation of the mode over time.
        """
        if t_span is None:
            t_span = (0, (self.n_time_steps - 1) * self.dt)

        times = np.linspace(t_span[0], t_span[1], n_frames)

        # Get the mode
        mode = self.phi[:, :, mode_idx]
        omega = self.omega[mode_idx]
        alpha = self.alpha[mode_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Define update function for animation
        def update(frame):
            ax.clear()
            t = times[frame]

            # Calculate mode at time t
            mode_t = mode * alpha * np.exp(omega * t)

            # For real-valued data, take the real part
            if np.isreal(self.data).all():
                mode_t = np.real(mode_t)

            # Plot mode
            im = ax.imshow(np.real(mode_t), cmap='RdBu_r')
            ax.set_title(f'Mode {mode_idx} at t = {t:.2f}')

            return [im]

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
        plt.tight_layout()

        return anim
        
    def compute_optimal_rank(self, max_rank=None, n_trials=5):
        """
        Compute the optimal rank for tensor decomposition by cross-validation.

        Parameters
        ----------
        max_rank : int, optional
            Maximum rank to consider. If None, uses min(dims)/2.
        n_trials : int, optional
            Number of cross-validation trials.

        Returns
        -------
        tuple
            Optimal ranks for each mode.
        dict
            Dictionary containing error information for different ranks.
        """
        if max_rank is None:
            max_rank = min(self.dims) // 2

        # Define ranks to test
        ranks_to_test = {}
        for mode in range(len(self.dims)):
            max_mode_rank = min(max_rank, self.dims[mode])
            ranks_to_test[mode] = np.linspace(1, max_mode_rank, 5, dtype=int)

        # Initialize error dictionary
        errors = {}

        # Test different combinations of ranks
        for trial in range(n_trials):
            print(f"Running trial {trial+1}/{n_trials}...")

            # Randomly sample 80% of the data for training
            train_mask = np.random.choice([True, False], size=self.n_time_steps, p=[0.8, 0.2])
            test_mask = ~train_mask

            train_data = self.data[train_mask]
            test_data = self.data[test_mask]

            # Test each rank combination
            for mode in range(len(self.dims)):
                for r in ranks_to_test[mode]:
                    # Create rank tuple
                    rank_tuple = [None] * len(self.dims)
                    rank_tuple[mode] = r
                    rank_tuple = tuple(rank_tuple)

                    try:
                        # Create TensorDMD model with current rank
                        tensor_dmd_model = TensorDMD(train_data, dt=self.dt, rank=rank_tuple)

                        # Reconstruct test times
                        test_times = np.arange(len(test_mask))[test_mask] * self.dt
                        predicted = tensor_dmd_model.predict(test_times)

                        # Calculate error
                        error = np.linalg.norm((test_data - predicted).reshape(-1)) / np.linalg.norm(test_data.reshape(-1))

                        # Store error
                        if rank_tuple not in errors:
                            errors[rank_tuple] = []
                        errors[rank_tuple].append(error)
                    except Exception as e:
                        print(f"Error with rank {rank_tuple}: {e}")

        # Compute average errors
        avg_errors = {k: np.mean(v) for k, v in errors.items() if len(v) == n_trials}

        # Find optimal rank
        if avg_errors:
            optimal_rank = min(avg_errors, key=avg_errors.get)
        else:
            optimal_rank = None

        return optimal_rank, avg_errors

    def is_mode_physical(self, mode_idx, threshold=0.1):
        """
        Determine if a mode is likely to be physical rather than noise.

        Parameters
        ----------
        mode_idx : int
            Index of the mode to check.
        threshold : float, optional
            Significance threshold.

        Returns
        -------
        bool
            True if the mode is likely physical, False otherwise.
        """
        # Get mode significance
        sig_df = self.mode_significance()
        mode_sig = sig_df.loc[sig_df['Mode'] == mode_idx, 'Significance'].values[0]

        # Check if the mode is significant
        is_significant = mode_sig > threshold

        # Check if the mode is well separated from other eigenvalues
        eigenvalues = self.Lambda
        current_eigenvalue = eigenvalues[mode_idx]

        distances = np.abs(eigenvalues - current_eigenvalue)
        distances[mode_idx] = float('inf')  # Exclude self
        min_distance = np.min(distances)

        is_well_separated = min_distance > 0.05

        # Check if the mode is spatially coherent (smooth)
        mode = self.phi[:, :, mode_idx]

        # Calculate a simple spatial coherence measure
        # For multi-dimensional tensors, reshape to 1D
        mode_flat = mode.reshape(-1)
        coherence = 1.0 - np.std(np.abs(mode_flat)) / np.mean(np.abs(mode_flat))

        is_coherent = coherence > 0.3

        # Combine criteria
        is_physical = is_significant and (is_well_separated or is_coherent)

        return is_physical
        
    def compute_svd_analysis(self):
        """
        Analyze the singular values of each unfolding of the tensor.

        Returns
        -------
        dict
            Dictionary containing singular values for each mode unfolding.
        """
        svd_info = {}

        for mode in range(len(self.dims)):
            # Unfold tensor along current mode
            X_unfolded = tl.unfold(self.data, mode=mode)

            # Compute SVD
            U, s, _ = np.linalg.svd(X_unfolded, full_matrices=False)

            # Store singular values
            svd_info[f'mode_{mode}'] = s

        return svd_info

    def plot_svd_analysis(self, figsize=(15, 5)):
        """
        Plot the singular values of each unfolding of the tensor.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the SVD analysis plots.
        """
        svd_info = self.compute_svd_analysis()

        # Create figure
        fig, axes = plt.subplots(1, len(self.dims), figsize=figsize)

        if len(self.dims) == 1:
            axes = [axes]  # Make it iterable

        for mode in range(len(self.dims)):
            ax = axes[mode]

            # Get singular values
            s = svd_info[f'mode_{mode}']

            # Plot singular values
            ax.semilogy(np.arange(1, len(s) + 1), s, 'o-', markersize=4)

            # Plot cumulative energy
            ax2 = ax.twinx()
            energy = s**2
            cumulative_energy = np.cumsum(energy) / np.sum(energy)
            ax2.plot(np.arange(1, len(s) + 1), cumulative_energy, 'r-', alpha=0.5)

            # Add thresholds
            thresholds = [0.9, 0.95, 0.99]
            for thresh in thresholds:
                idx = np.argmax(cumulative_energy >= thresh)
                ax2.axhline(thresh, color='r', linestyle='--', alpha=0.3)
                ax2.text(len(s) * 0.7, thresh, f'{thresh:.0%}: {idx+1}',
                         verticalalignment='bottom', color='r')

            # Set labels and title
            ax.set_xlabel('Index')
            ax.set_ylabel('Singular Value')
            ax2.set_ylabel('Cumulative Energy', color='r')
            ax.set_title(f'Mode {mode} Unfolding')
            ax.grid(True)

        plt.tight_layout()
        return fig
        
    @staticmethod
    def generate_example_tensor(n_time=50, nx=20, ny=20, freq1=0.1, freq2=0.05, noise_level=0.05):
        """
        Generate a synthetic tensor with oscillatory patterns for testing.

        Parameters
        ----------
        n_time : int, optional
            Number of time steps.
        nx, ny : int, optional
            Spatial dimensions.
        freq1, freq2 : float, optional
            Frequencies of oscillatory patterns.
        noise_level : float, optional
            Level of noise to add.

        Returns
        -------
        numpy.ndarray
            Synthetic tensor data.
        """
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 4*np.pi, n_time)

        X, Y = np.meshgrid(x, y)

        # Create a tensor with oscillatory patterns
        tensor = np.zeros((n_time, nx, ny))

        for i, time in enumerate(t):
            # First spatial pattern oscillating at freq1
            pattern1 = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y) * np.sin(freq1*time)

            # Second spatial pattern oscillating at freq2
            pattern2 = np.sin(4*np.pi*X) * np.sin(4*np.pi*Y) * np.cos(freq2*time)

            # Combine patterns
            tensor[i] = pattern1 + pattern2

        # Add noise
        tensor += noise_level * np.random.randn(*tensor.shape)

        return tensor

    def __str__(self):
        """String representation of the TensorDMD object."""
        return (f"TensorDMD(dims={self.dims}, dt={self.dt}, "
                f"rank={self.rank}, truncate_rank={self.truncate_rank})")

    def __repr__(self):
        """Representation of the TensorDMD object."""
        return self.__str__()
        
    def compute_svd_analysis(self):
        """
        Analyze the singular values of each unfolding of the tensor.

        Returns
        -------
        dict
            Dictionary containing singular values for each mode unfolding.
        """
        svd_info = {}

        for mode in range(len(self.dims)):
            # Unfold tensor along current mode
            X_unfolded = tl.unfold(self.data, mode=mode)

            # Compute SVD
            U, s, _ = np.linalg.svd(X_unfolded, full_matrices=False)

            # Store singular values
            svd_info[f'mode_{mode}'] = s

        return svd_info

    def plot_svd_analysis(self, figsize=(15, 5)):
        """
        Plot the singular values of each unfolding of the tensor.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the SVD analysis plots.
        """
        svd_info = self.compute_svd_analysis()

        # Create figure
        fig, axes = plt.subplots(1, len(self.dims), figsize=figsize)

        if len(self.dims) == 1:
            axes = [axes]  # Make it iterable

        for mode in range(len(self.dims)):
            ax = axes[mode]

            # Get singular values
            s = svd_info[f'mode_{mode}']

            # Plot singular values
            ax.semilogy(np.arange(1, len(s) + 1), s, 'o-', markersize=4)

            # Plot cumulative energy
            ax2 = ax.twinx()
            energy = s**2
            cumulative_energy = np.cumsum(energy) / np.sum(energy)
            ax2.plot(np.arange(1, len(s) + 1), cumulative_energy, 'r-', alpha=0.5)

            # Add thresholds
            thresholds = [0.9, 0.95, 0.99]
            for thresh in thresholds:
                idx = np.argmax(cumulative_energy >= thresh)
                ax2.axhline(thresh, color='r', linestyle='--', alpha=0.3)
                ax2.text(len(s) * 0.7, thresh, f'{thresh:.0%}: {idx+1}',
                         verticalalignment='bottom', color='r')

            # Set labels and title
            ax.set_xlabel('Index')
            ax.set_ylabel('Singular Value')
            ax2.set_ylabel('Cumulative Energy', color='r')
            ax.set_title(f'Mode {mode} Unfolding')
            ax.grid(True)

        plt.tight_layout()
        return fig
        
    def plot_mode_contributions(self, n_modes=10, figsize=(10, 6)):
        """
        Plot the contribution of each mode to the overall dynamics as a pie chart.

        Parameters
        ----------
        n_modes : int, optional
            Number of modes to include in the plot.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the contribution plot.
        """
        # Calculate mode contribution based on amplitude and eigenvalue magnitude
        amplitudes = np.abs(self.alpha)
        magnitudes = np.abs(self.Lambda)

        # For growing modes (|λ| > 1), scale by the growth over the time span
        final_time = (self.n_time_steps - 1) * self.dt
        for i, mag in enumerate(magnitudes):
            if mag > 1:
                magnitudes[i] = mag ** final_time

        # Calculate contribution as amplitude times magnitude
        contributions = amplitudes * magnitudes
        total_contribution = np.sum(contributions)
        normalized_contributions = contributions / total_contribution if total_contribution > 0 else contributions

        # Sort by contribution
        sorted_indices = np.argsort(normalized_contributions)[::-1]

        # Select top n_modes
        n_modes = min(n_modes, len(sorted_indices))
        top_indices = sorted_indices[:n_modes]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create pie chart
        labels = [f'Mode {idx}\n{normalized_contributions[idx]:.2%}' for idx in top_indices]
        wedges, texts = ax.pie(normalized_contributions[top_indices],
                                labels=None,
                                autopct=None,
                                startangle=90,
                                counterclock=False,
                                wedgeprops={'edgecolor': 'w', 'linewidth': 1})

        # Create legend
        ax.legend(wedges, labels, title="Mode Contributions",
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        ax.set_title('DMD Mode Contributions to Overall Dynamics')

        plt.tight_layout()
        return fig
        
    def plot_3d_spectrum(self, figsize=(12, 10)):
        """
        Create a 3D plot of the DMD spectrum with eigenvalues, growth rates, and frequencies.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the 3D spectrum plot.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Extract properties
        growth_rates = np.real(self.omega)
        frequencies = np.imag(self.omega) / (2 * np.pi)  # Convert to cycles per time unit
        amplitudes = np.abs(self.alpha)
        eigenvalue_magnitudes = np.abs(self.Lambda)

        # Normalize amplitudes for point size
        normalized_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes

        # Create scatter plot
        scatter = ax.scatter(growth_rates, frequencies, eigenvalue_magnitudes,
                            s=normalized_amplitudes*200,
                            c=eigenvalue_magnitudes,
                            cmap='viridis',
                            alpha=0.7)

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Eigenvalue Magnitude |λ|')

        # Set labels and title
        ax.set_xlabel('Growth Rate')
        ax.set_ylabel('Frequency (cycles per time unit)')
        ax.set_zlabel('Eigenvalue Magnitude |λ|')
        ax.set_title('3D DMD Spectrum')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Annotate significant modes
        significant_modes = np.argsort(normalized_amplitudes)[-5:]
        for i in significant_modes:
            ax.text(growth_rates[i], frequencies[i], eigenvalue_magnitudes[i], f'{i}', fontsize=10)

        return fig

    def plot_continuous_spectrum(self, figsize=(10, 8), size_factor=100):
        """
        Plot the continuous-time DMD spectrum.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        size_factor : float, optional
            Scaling factor for the size of the points.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the continuous-time spectrum plot.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot eigenvalues
        amplitudes = np.abs(self.alpha)
        normalized_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes

        scatter = ax.scatter(self.omega.real, self.omega.imag,
                             s=normalized_amplitudes*size_factor,
                             c=np.abs(self.omega),
                             cmap='plasma',
                             alpha=0.7)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Magnitude |ω|')

        # Set labels and title
        ax.set_xlabel('Growth Rate (Real(ω))')
        ax.set_ylabel('Frequency (Imag(ω))')
        ax.set_title('DMD Continuous-Time Spectrum')

        # Add grid
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)

        # Annotate significant modes
        significant_modes = np.argsort(normalized_amplitudes)[-5:]
        for i in significant_modes:
            ax.annotate(f'{i}',
                        (self.omega[i].real, self.omega[i].imag),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=10)

        return fig

    def plot_mode_amplitudes(self, figsize=(10, 6), n_modes=None):
        """
        Plot the amplitudes of the DMD modes.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        n_modes : int, optional
            Number of modes to plot. If None, plots all modes.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the amplitude plot.
        """
        amplitudes = np.abs(self.alpha)

        if n_modes is None:
            n_modes = len(amplitudes)
        else:
            n_modes = min(n_modes, len(amplitudes))

        # Sort indices by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1][:n_modes]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(np.arange(n_modes), amplitudes[sorted_indices])

        # Color bars by eigenvalue magnitude
        magnitudes = np.abs(self.Lambda[sorted_indices])
        normalized_magnitudes = magnitudes / np.max(magnitudes) if np.max(magnitudes) > 0 else magnitudes

        # Create colormap
        cmap = plt.cm.viridis

        for i, bar in enumerate(bars):
            bar.set_color(cmap(normalized_magnitudes[i]))

        # Set labels and title
        ax.set_xlabel('Mode Index (sorted by amplitude)')
        ax.set_ylabel('Amplitude')
        ax.set_title('DMD Mode Amplitudes')

        # Add mode indices as x-tick labels
        ax.set_xticks(np.arange(n_modes))
        ax.set_xticklabels([f'{idx}' for idx in sorted_indices])

        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_mode_frequencies(self, figsize=(10, 6), n_modes=None):
        """
        Plot the frequencies of the DMD modes.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        n_modes : int, optional
            Number of modes to plot. If None, plots all modes.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the frequency plot.
        """
        frequencies = np.abs(np.imag(self.omega) / (2 * np.pi))  # Convert to cycles per time unit
        amplitudes = np.abs(self.alpha)

        # Sort by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1]

        if n_modes is None:
            n_modes = len(frequencies)
        else:
            n_modes = min(n_modes, len(frequencies))

        sorted_indices = sorted_indices[:n_modes]

        fig, ax = plt.subplots(figsize=figsize)

        # Create scatter plot with size proportional to amplitude
        normalized_amplitudes = amplitudes[sorted_indices] / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes[sorted_indices]

        scatter = ax.scatter(sorted_indices, frequencies[sorted_indices],
                             s=normalized_amplitudes*200,
                             c=normalized_amplitudes,
                             cmap='viridis',
                             alpha=0.7)

        # Connect points with line
        ax.plot(np.arange(n_modes), frequencies[sorted_indices], 'k-', alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Normalized Amplitude')

        # Set labels and title
        ax.set_xlabel('Mode Index (sorted by amplitude)')
        ax.set_ylabel('Frequency (cycles per time unit)')
        ax.set_title('DMD Mode Frequencies')

        # Add mode indices as x-tick labels
        ax.set_xticks(np.arange(n_modes))
        ax.set_xticklabels([f'{idx}' for idx in sorted_indices])

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_growth_rates(self, figsize=(10, 6), n_modes=None):
        """
        Plot the growth rates of the DMD modes.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        n_modes : int, optional
            Number of modes to plot. If None, plots all modes.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the growth rate plot.
        """
        growth_rates = np.real(self.omega)
        amplitudes = np.abs(self.alpha)

        # Sort by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1]

        if n_modes is None:
            n_modes = len(growth_rates)
        else:
            n_modes = min(n_modes, len(growth_rates))

        sorted_indices = sorted_indices[:n_modes]

        fig, ax = plt.subplots(figsize=figsize)

        # Create bar chart
        bars = ax.bar(np.arange(n_modes), growth_rates[sorted_indices])

        # Color bars by sign of growth rate
        for i, bar in enumerate(bars):
            if growth_rates[sorted_indices[i]] >= 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')

        # Set labels and title
        ax.set_xlabel('Mode Index (sorted by amplitude)')
        ax.set_ylabel('Growth Rate')
        ax.set_title('DMD Mode Growth Rates')

        # Add horizontal line at y=0
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)

        # Add mode indices as x-tick labels
        ax.set_xticks(np.arange(n_modes))
        ax.set_xticklabels([f'{idx}' for idx in sorted_indices])

        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_reconstruction_error(self, figsize=(10, 6)):
        """
        Plot the reconstruction error as a function of time.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the error plot.
        """
        # Reconstruct the data
        reconstructed = self.reconstruct()

        # Calculate error for each time step
        errors = np.zeros(self.n_time_steps)
        for t in range(self.n_time_steps):
            norm_orig = np.linalg.norm(self.data[t].reshape(-1))
            if norm_orig > 1e-10:  # Avoid division by zero
                errors[t] = np.linalg.norm((self.data[t] - reconstructed[t]).reshape(-1)) / norm_orig
            else:
                errors[t] = np.linalg.norm((self.data[t] - reconstructed[t]).reshape(-1))

        times = np.arange(self.n_time_steps) * self.dt

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, errors, 'o-', markersize=4)

        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Relative Error')
        ax.set_title('Tensor DMD Reconstruction Error')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Calculate mean error
        mean_error = np.mean(errors)
        ax.axhline(mean_error, color='red', linestyle='--', alpha=0.7,
                  label=f'Mean Error: {mean_error:.4f}')

        ax.legend()
        plt.tight_layout()

        return fig

    def plot_reconstruction_comparison(self, time_indices=None, n_snapshots=3, figsize=(15, 8)):
        """
        Plot comparison between original and reconstructed snapshots.

        Parameters
        ----------
        time_indices : list, optional
            Indices of time steps to visualize. If None, selects equidistant snapshots.
        n_snapshots : int, optional
            Number of snapshots to visualize if time_indices is None.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the comparison plots.
        """
        if time_indices is None:
            # Select equidistant snapshots
            time_indices = np.linspace(0, self.n_time_steps-1, n_snapshots, dtype=int)
        else:
            n_snapshots = len(time_indices)

        # Reconstruct the data
        reconstructed = self.reconstruct()

        # Create figure
        fig, axes = plt.subplots(2, n_snapshots, figsize=figsize)

        if n_snapshots == 1:
            axes = axes.reshape(2, 1)  # Make it 2D array

        for i, t_idx in enumerate(time_indices):
            if t_idx < self.n_time_steps:
                # Original
                im1 = axes[0, i].imshow(self.data[t_idx], cmap='viridis')
                axes[0, i].set_title(f'Original, t={t_idx * self.dt:.2f}')
                plt.colorbar(im1, ax=axes[0, i])

                # Reconstructed
                im2 = axes[1, i].imshow(reconstructed[t_idx], cmap='viridis')
                axes[1, i].set_title(f'Reconstructed, t={t_idx * self.dt:.2f}')
                plt.colorbar(im2, ax=axes[1, i])

        plt.tight_layout()
        return fig

    def get_core_tensor(self):
        """
        Get the core tensor from the Tucker decomposition.

        Returns
        -------
        numpy.ndarray
            Core tensor.
        """
        return self.core

    def get_factor_matrices(self):
        """
        Get the factor matrices from the Tucker decomposition.

        Returns
        -------
        list
            List of factor matrices for each mode.
        """
        return self.factors

    def reconstruct_tucker(self):
        """
        Reconstruct the data using only the Tucker decomposition (without DMD).

        Returns
        -------
        numpy.ndarray
            Reconstructed tensor using Tucker decomposition.
        """
        # Reconstruct X1 using the core tensor and factor matrices
        reconstructed = tl.tucker_tensor.tucker_to_tensor((self.core, self.factors))

        # Pad to match original data length
        padded = np.zeros_like(self.data)
        padded[:-1] = reconstructed
        padded[-1] = reconstructed[-1]  # Repeat last frame

        return padded

    def compute_mode_energy(self):
        """
        Compute the energy contribution of each DMD mode.

        Returns
        -------
        numpy.ndarray
            Energy contribution of each mode.
        """
        amplitudes = np.abs(self.alpha)
        energies = amplitudes**2 / np.sum(amplitudes**2)
        return energies

    def visualize_modes(self, mode_indices=None, n_modes=3, figsize=(15, 5)):
        """
        Visualize the spatial structure of selected DMD modes with debug information.

        Parameters
        ----------
        mode_indices : list, optional
            Indices of modes to visualize. If None, visualizes the top modes by amplitude.
        n_modes : int, optional
            Number of modes to visualize if mode_indices is None.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the visualizations.
        """
        if mode_indices is None:
            # Get the top modes by amplitude
            mode_indices = np.argsort(np.abs(self.alpha))[-n_modes:][::-1]
        else:
            n_modes = len(mode_indices)

        # DEBUG: Print overall mode and dimension info.
        print("DEBUG: self.dims =", self.dims)
        print("DEBUG: self.phi.shape =", self.phi.shape)

        # Create figure with n_modes columns and 2 rows (real and imaginary parts)
        fig, axes = plt.subplots(2, n_modes, figsize=(figsize[0], figsize[1] * 2))
        if n_modes == 1:
            axes = axes.reshape(2, 1)  # Ensure a 2D array for axes

        for i, mode_idx in enumerate(mode_indices):
            if i < n_modes:
                print(f"\nDEBUG: Visualizing mode index: {mode_idx}")
                # Correct extraction: use the third axis for mode index.
                current_mode = self.phi[:, :, mode_idx]
                print("DEBUG: current_mode shape (should be spatial dimensions):", current_mode.shape)

                # Check that the current_mode matches the expected spatial dimensions.
                target_shape = self.dims[1:]
                print("DEBUG: Expected spatial shape (self.dims[1:]):", target_shape)

                if current_mode.shape != target_shape:
                    print("WARNING: Mode shape does not match expected spatial dimensions.")

                # Plot the real part of the mode.
                im_real = axes[0, i].imshow(np.real(current_mode), cmap='RdBu_r')
                axes[0, i].set_title(f'Mode {mode_idx} (Real Part)\nλ={self.Lambda[mode_idx]:.2f}')
                plt.colorbar(im_real, ax=axes[0, i])

                # Plot the imaginary part of the mode.
                im_imag = axes[1, i].imshow(np.imag(current_mode), cmap='RdBu_r')
                axes[1, i].set_title(f'Mode {mode_idx} (Imaginary Part)')
                plt.colorbar(im_imag, ax=axes[1, i])

        plt.tight_layout()
        return fig