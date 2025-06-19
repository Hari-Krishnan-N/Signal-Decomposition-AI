import librosa
import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from PIL import Image

@dataclass
class VMDResult:
    """Container for VMD decomposition results."""
    modes: np.ndarray  # Decomposed modes in time domain
    modes_hat: np.ndarray  # Fourier-domain representation
    center_frequencies: np.ndarray  # Center frequencies of modes
    n_iterations: int  # Number of iterations performed
    convergence_metric: float  # Final convergence metric value

class GPU_VMD:
    """Variational Mode Decomposition implemented in PyTorch with GPU support."""
    
    def __init__(
        self,
        alpha: float,
        tau: float,
        n_modes: int,
        dc_component: bool = False,
        init_method: int = 1,
        tolerance: float = 1e-6,
        max_iterations: int = 500,
        device: Optional[str] = None
    ):
        """
        Initialize VMD parameters.
        
        Parameters:
        -----------
        alpha : float
            Bandwidth constraint parameter
        tau : float
            Noise-tolerance parameter (Lagrangian multiplier)
        n_modes : int
            Number of modes to extract
        dc_component : bool
            Whether to force first mode to be DC (zero frequency)
        init_method : int
            Frequency initialization method:
            0: All zeros
            1: Linear spacing
            2: Random logarithmic spacing
        tolerance : float
            Convergence tolerance
        max_iterations : int
            Maximum number of iterations
        device : Optional[str]
            Computing device ('cuda' or 'cpu'). If None, automatically selects.
        """
        self.alpha = alpha
        self.tau = tau
        self.n_modes = n_modes
        self.dc_component = dc_component
        self.init_method = init_method
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert parameters to tensors
        self._setup_parameters()
    
    def _setup_parameters(self) -> None:
        """Initialize parameters as tensors on the correct device."""
        self.device = torch.device(self.device)
        self.alpha_tensor = torch.full((self.n_modes,), self.alpha, device=self.device)
    
    def _init_frequencies(self, fs: torch.Tensor) -> torch.Tensor:
        """Initialize center frequencies based on the chosen method."""
        if self.init_method == 1:
            omega = torch.linspace(0, 0.5, self.n_modes, device=self.device)
        elif self.init_method == 2:
            omega = torch.sort(
                torch.exp(
                    torch.log(fs) + 
                    (torch.log(torch.tensor(0.5, device=self.device)) - torch.log(fs)) * 
                    torch.rand(self.n_modes, device=self.device)
                )
            )[0]
        else:
            omega = torch.zeros(self.n_modes, device=self.device)
            
        if self.dc_component:
            omega[0] = 0
            
        return omega
    
    @staticmethod
    def _mirror_signal(signal: torch.Tensor) -> torch.Tensor:
        """Mirror the signal at the boundaries."""
        ltemp = len(signal) // 2
        mirrored = torch.empty(len(signal) + 2 * ltemp, device=signal.device)
        mirrored[:ltemp] = torch.flip(signal[:ltemp], dims=[0])
        mirrored[ltemp:ltemp + len(signal)] = signal
        mirrored[ltemp + len(signal):] = torch.flip(signal[-ltemp:], dims=[0])
        return mirrored
    
    def decompose(self, signal: np.ndarray, alpa: float) -> VMDResult:
        """
        Decompose the input signal into modes.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal (1D array)
            
        Returns:
        --------
        VMDResult
            Dataclass containing decomposition results
        """
        
        self.alpha_tensor = torch.full((self.n_modes,), alpa, device=self.device)
        # Ensure signal length is even
        if len(signal) % 2:
            signal = signal[:-1]
            
        # Convert to tensor and mirror
        f = torch.tensor(signal, dtype=torch.float32, device=self.device)
        fs = torch.tensor(1.0 / len(f), device=self.device)
        f_mirrored = self._mirror_signal(f)
        
        # Setup frequency domain
        T = len(f_mirrored)
        freqs = torch.arange(T, device=self.device) / T - 0.5 - (1 / T)
        
        # Initialize frequency domain signal
        f_hat = torch.fft.fftshift(torch.fft.fft(f_mirrored))
        f_hat_plus = f_hat.clone()
        f_hat_plus[:T // 2] = 0
        
        # Initialize modes and frequencies
        omega = self._init_frequencies(fs)
        u_prev = torch.zeros((T, self.n_modes), dtype=torch.cfloat, device=self.device)
        u_curr = u_prev.clone()
        lambda_acc = torch.zeros(T, dtype=torch.cfloat, device=self.device)
        
        # Main loop
        n_iter = 0
        conv_metric = float('inf')
        
        while conv_metric > self.tolerance and n_iter < self.max_iterations:
            u_prev.copy_(u_curr)
            
            # Update first mode
            sum_uk = torch.sum(u_prev, dim=1) - u_prev[:, 0]
            u_curr[:, 0] = (f_hat_plus - sum_uk - lambda_acc / 2) / (1 + self.alpha_tensor[0] * (freqs - omega[0]) ** 2)
            
            if not self.dc_component:
                omega[0] = self._update_omega(u_curr[:, 0], freqs, T)
            
            # Update remaining modes
            for k in range(1, self.n_modes):
                sum_uk += u_curr[:, k-1] - u_prev[:, k]
                u_curr[:, k] = (f_hat_plus - sum_uk - lambda_acc / 2) / (1 + self.alpha_tensor[k] * (freqs - omega[k]) ** 2)
                omega[k] = self._update_omega(u_curr[:, k], freqs, T)
            
            # Update Lagrange multiplier
            lambda_acc += self.tau * (torch.sum(u_curr, dim=1) - f_hat_plus)
            
            # Check convergence
            conv_metric = torch.sum(torch.abs(u_curr - u_prev) ** 2).item() / T
            n_iter += 1
        
        # Post-process results
        modes, modes_hat = self._post_process(u_curr, T)
        
        return VMDResult(
            modes=modes.cpu().numpy(),
            modes_hat=modes_hat.cpu().numpy(),
            center_frequencies=omega.cpu().numpy(),
            n_iterations=n_iter,
            convergence_metric=conv_metric
        )
    
    @staticmethod
    def _update_omega(u_k: torch.Tensor, freqs: torch.Tensor, T: int) -> torch.Tensor:
        """Update center frequency for a mode."""
        weights = torch.abs(u_k[T // 2:T]) ** 2
        return torch.sum(freqs[T // 2:T] * weights) / torch.sum(weights)
    
    def _post_process(
        self, u_curr: torch.Tensor, T: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Post-process modes after convergence."""
        # Reconstruct symmetric spectrum
        u_hat = torch.zeros_like(u_curr)
        u_hat[T // 2:T, :] = u_curr[T // 2:T, :]
        u_hat[1:T // 2, :] = torch.conj(torch.flip(u_curr[T // 2 + 1:T, :], dims=[0]))
        u_hat[0, :] = torch.conj(u_hat[-1, :])
        
        # Convert to time domain
        u = torch.zeros((self.n_modes, T), device=self.device)
        for k in range(self.n_modes):
            u[k] = torch.real(torch.fft.ifft(torch.fft.ifftshift(u_hat[:, k])))
        
        # Trim mirrored parts
        u = u[:, T // 4:3 * T // 4]
        
        # Compute final frequency domain representation
        u_hat_final = torch.zeros((u.shape[1], self.n_modes), dtype=torch.cfloat, device=self.device)
        for k in range(self.n_modes):
            u_hat_final[:, k] = torch.fft.fftshift(torch.fft.fft(u[k]))
            
        return u, u_hat_final

def pad_signal(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """
    Pad or trim reconstructed signal to match original length.
    
    Parameters:
    -----------
    original : np.ndarray
        Original signal
    reconstructed : np.ndarray
        Reconstructed signal
        
    Returns:
    --------
    np.ndarray
        Padded/trimmed reconstructed signal
    """
    if len(reconstructed) < len(original):
        return np.pad(reconstructed, (0, len(original) - len(reconstructed)))
    return reconstructed[:len(original)]





def save_modes_and_reconstruct(modes, sample_rate, original_signal, base_filename, output_dir='VMD/Outputs', exclude_modes=None):
    """
    Save the decomposed VMD modes and reconstructed signal.
    
    Parameters:
    -----------
    modes : numpy.ndarray
        The decomposed modes from VMD (shape: [n_modes, signal_length])
    sample_rate : int
        Sampling rate of the original audio
    original_signal : numpy.ndarray
        The original input signal
    base_filename : str
        Base filename to use for saving outputs
    output_dir : str
        Directory to save the output files (default: 'processed_audio')
    exclude_modes : list of int, optional
        Indices of modes to exclude from the reconstructed signal
    """
    # Create output directories if they don't exist
    modes_dir = os.path.join(output_dir, 'IMFs')
    plots_dir = os.path.join(output_dir, 'Plots')
    reconstructed_dir = os.path.join(output_dir, 'Reconstructed')
    spectrograms_dir = os.path.join(output_dir, 'Spectrograms')
    
    for directory in [modes_dir, plots_dir, reconstructed_dir, spectrograms_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Save individual modes as WAV files
    for i, mode in enumerate(modes):
        mode_filename = os.path.join(modes_dir, f"{base_filename}_mode_{i+1}.wav")
        # Normalize mode to prevent clipping
        normalized_mode = mode / np.max(np.abs(mode))
        wavfile.write(mode_filename, sample_rate, normalized_mode.astype(np.float32))
        
        S_stft = librosa.stft(normalized_mode)
        S_stft_db = librosa.amplitude_to_db(np.abs(S_stft), ref=np.max)

        # ---- Generate Mel Spectrogram ----
        S_mel = librosa.feature.melspectrogram(y=normalized_mode, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=512)
        S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
        
        # Save and resize STFT Spectrogram
        stft_spectrogram_filename = os.path.join(spectrograms_dir, f"{base_filename}_mode_{i+1}_stft.png")
        save_resized_spectrogram(S_stft_db, sample_rate, y_axis='log', filename=stft_spectrogram_filename)

        # Save and resize Mel Spectrogram
        mel_spectrogram_filename = os.path.join(spectrograms_dir, f"{base_filename}_mode_{i+1}_mel.png")
        save_resized_spectrogram(S_mel_db, sample_rate, y_axis='mel', filename=mel_spectrogram_filename)
        
        
    
    # Reconstruct signal by summing all modes except the excluded ones
    if exclude_modes is None:
        reconstructed_signal = np.sum(modes, axis=0)
    else:
        indices = []
        for i in range(len(modes)):
            if i not in exclude_modes:
                indices.append(i)

        reconstructed_signal = np.sum(modes[indices], axis=0)

    
    # Normalize reconstructed signal
    reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))
    
    # Save reconstructed signal
    reconstructed_filename = os.path.join(reconstructed_dir, f"{base_filename}_reconstructed.wav")
    wavfile.write(reconstructed_filename, sample_rate, reconstructed_signal.astype(np.float32))

    
    
    plot_vmd_modes_with_spectrum(np.arange(len(original_signal)), original_signal, reconstructed_signal, modes, sample_rate, plots_dir, base_filename, include_spectrogram=True, signal_plot_len=2000)
    
    # Calculate and return reconstruction error
    reconstruction_error = np.mean((original_signal - reconstructed_signal) ** 2)
    
    return reconstruction_error


def plot_vmd_modes_with_spectrum(t, signal, reconstructed_signal, modes, sampling_rate, plots_dir, base_filename, include_spectrogram=True, signal_plot_len=None):
    """ 
    Plot the original signal, decomposed modes in time domain and their frequency spectra.
    Optionally include spectrograms.
    
    Parameters:
    t (numpy array): Time values
    signal (numpy array): Original signal
    modes (list of numpy arrays): Decomposed modes
    sampling_rate (float): Sampling rate of the signal (Hz)
    include_spectrogram (bool): Whether to include spectrograms in the visualization
    """
    num_modes = len(modes)
    if signal_plot_len is None:
        signal_plot_len=signal.shape[0]
    
    # Adjust subplot configuration based on spectrogram option
    if include_spectrogram:
        fig, axes = plt.subplots(num_modes + 2, 3, figsize=(36, 3 * (num_modes + 2)))
        plot_columns = 3
    else:
        fig, axes = plt.subplots(num_modes + 2, 2, figsize=(24, 3 * (num_modes + 2)))
        plot_columns = 2
    
    # Plot original signal - Time Domain
    axes[0, 0].plot(t[:signal_plot_len], signal[:signal_plot_len], label="Original Signal", color='black', linewidth=1)
    axes[0, 0].set_title("Original Signal - Time Domain")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)
    # axes[0, 0].legend()
    
    # Original signal frequency spectrum
    freq = np.fft.rfftfreq(len(signal), d=1/sampling_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    axes[0, 1].plot(freq, spectrum, label="Original Signal Frequency Spectrum", color='black', linewidth=1)
    axes[0, 1].set_title("Original Signal - Frequency Domain")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True)
    # axes[0, 1].legend()
    
    # Original signal spectrogram (if enabled)
    if include_spectrogram:
        S = librosa.stft(signal)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(S_db, sr=sampling_rate, x_axis='time', y_axis='log', 
                                 cmap='magma', ax=axes[0, 2])
        axes[0, 2].set_title("Original Signal Spectrogram")
        
    # Plot reconstructed signal - Time Domain
    axes[1, 0].plot(t[:signal_plot_len], reconstructed_signal[:signal_plot_len], label="Reconstructed Signal", color='red', linewidth=1)
    axes[1, 0].set_title("Reconstructed Signal - Time Domain")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True)
    # axes[1, 0].legend()
    
    # reconstructed_signal frequency spectrum
    freq = np.fft.rfftfreq(len(reconstructed_signal), d=1/sampling_rate)
    spectrum = np.abs(np.fft.rfft(reconstructed_signal))
    axes[1, 1].plot(freq, spectrum, label="Reconstructed Signal Frequency Spectrum", color='red', linewidth=1)
    axes[1, 1].set_title("Reconstructed Signal - Frequency Domain")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(True)
    # axes[1, 1].legend()
    
    # reconstructed_signal spectrogram (if enabled)
    if include_spectrogram:
        S = librosa.stft(reconstructed_signal)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(S_db, sr=sampling_rate, x_axis='time', y_axis='log', 
                                 cmap='magma', ax=axes[1, 2])
        axes[1, 2].set_title("Reconstructed Signal Spectrogram")
    
    # Iterate over each mode to plot time-domain and frequency-domain
    for i, mode in enumerate(modes):
        # Plot time-domain waveform
        axes[i+2, 0].plot(t[:signal_plot_len], mode[:signal_plot_len], label=f"IMF{i+1}", color='blue', linewidth=1)
        axes[i+2, 0].set_title(f"IMF{i+1} - Time Domain")
        axes[i+2, 0].set_xlabel("Time (s)")
        axes[i+2, 0].set_ylabel("Amplitude")
        axes[i+2, 0].grid(True)
        # axes[i+2, 0].legend()
        
        # Compute frequency spectrum
        freq = np.fft.rfftfreq(len(mode), d=1/sampling_rate)
        spectrum = np.abs(np.fft.rfft(mode))
        axes[i+2, 1].plot(freq, spectrum, label=f"Frequency Spectrum of IMF{i+1}", color='green', linewidth=1)
        axes[i+2, 1].set_title(f"IMF{i+1} - Frequency Domain")
        axes[i+2, 1].set_xlabel("Frequency (Hz)")
        axes[i+2, 1].set_ylabel("Amplitude")
        axes[i+2, 1].grid(True)
        # axes[i+2, 1].legend()
        
        # Plot mode spectrogram (if enabled)
        if include_spectrogram:
            S_mode = librosa.stft(mode)
            S_mode_db = librosa.amplitude_to_db(np.abs(S_mode), ref=np.max)
            librosa.display.specshow(S_mode_db, sr=sampling_rate, x_axis='time', y_axis='log', 
                                     cmap='magma', ax=axes[i+2, 2])
            axes[i+2, 2].set_title(f"IMF{i+1} Spectrogram")
    
    plt.tight_layout()
    plot_filename = os.path.join(plots_dir, f"{base_filename}_analysis.png")
    plt.savefig(plot_filename)
    plt.close()
    
    
    
# ---- Function to Plot, Resize, and Save Spectrogram Without Borders ----
def save_resized_spectrogram(S_db, sr, y_axis, filename, size=(512, 512)):
    # Create a figure with no padding and no frame
    fig = Figure(figsize=(5, 4), dpi=300, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure area (left, bottom, width, height)
    
    # Plot the spectrogram
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis=y_axis, cmap='gray', ax=ax)
    
    # Remove axes and labels
    ax.axis('off')
    
    # Draw the canvas and convert to NumPy array
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    
    # Convert to PIL Image, resize, and save
    im = Image.fromarray(rgba)
    im_resized = im.resize(size, Image.LANCZOS)  # LANCZOS for high-quality downsampling
    im_resized.save(filename)
    plt.close(fig)

    
    
    
    
    

