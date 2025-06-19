
import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

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
    
    def decompose(self, signal: np.ndarray) -> VMDResult:
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

