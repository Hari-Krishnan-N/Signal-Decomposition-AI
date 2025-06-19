import os 
import torch
import numpy as np 
from scipy.io import wavfile
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from typing import List, Tuple

class EmpricalModeDecomposition:
    """
    A class for performing Empirical Mode Decomposition (EMD) on a given signal.
    EMD decomposes a signal into Intrinsic Mode Functions (IMFs) and a residual component.

    Attributes:
        max_imfs (int): Maximum number of IMFs to extract from the signal.
        max_iter (int): Maximum number of iterations for the sifting process.
        tol (float): Tolerance level for stopping criterion on the residual. Default is 1e-8.
        device (torch.device): Device to perform computations (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, max_imfs: int = 10, max_iter: int = 128, tol: float = 1e-5, device: torch.device = 'cpu'):
        """
        Initializes the EmpiricalModeDecomposition class.

        Args:
            max_imfs (int): Maximum number of IMFs to extract. Default is 10.
            max_iter (int): Maximum number of iterations for the sifting process. Default is 128.
            tol (float): Tolerance level for stopping criterion on the residual. Default is 1e-8.
            device (torch.device): Device to perform computations. Default is 'cpu'.
        """
        super(EmpricalModeDecomposition, self).__init__()
        self.max_imfs = max_imfs
        self.max_iter = max_iter 
        self.device = device
        self.tol = tol

    def count_zero_crossings(self, signal: torch.tensor) -> int:
        """
        Counts the number of zero-crossings in the signal.

        Args:
            signal (torch.tensor): Input signal.

        Returns:
            int: Number of zero-crossings in the signal.
        """
        return torch.sum(torch.diff(torch.sign(signal)) != 0)
    
    def sift(self, residual: torch.tensor) -> torch.tensor:
        """
        Performs the sifting process to extract one Intrinsic Mode Function (IMF).

        Args:
            residual (torch.tensor): The residual signal to be decomposed.

        Returns:
            torch.tensor: Extracted IMF from the residual signal.
        """
        h = residual.clone()

        for i in range(self.max_iter):
            # Find local maxima and minima
            maxima = torch.where((h[:-2] < h[1:-1]) & (h[1:-1] > h[2:]))[0] + 1
            minima = torch.where((h[:-2] > h[1:-1]) & (h[1:-1] < h[2:]))[0] + 1

            # If insufficient extrema, stop the process
            if len(maxima) < 2 or len(minima) < 2:
                break
            
            # Convert tensors to numpy for interpolation
            h_numpy = h.cpu().numpy().copy()
            maxima_numpy = maxima.cpu().numpy().copy()
            minima_numpy = minima.cpu().numpy().copy()

            # Compute upper and lower envelopes using cubic spline interpolation
            upper_env = CubicSpline(maxima_numpy, h_numpy[maxima_numpy])(np.arange(len(h_numpy)))
            lower_env = CubicSpline(minima_numpy, h_numpy[minima_numpy])(np.arange(len(h_numpy)))

            # Convert back to tensors
            h = torch.from_numpy(h_numpy).to(self.device)
            upper_env = torch.from_numpy(upper_env).to(self.device)
            lower_env = torch.from_numpy(lower_env).to(self.device)

            # Compute mean envelope and update signal
            mean_env = (upper_env + lower_env) / 2
            new_h = h - mean_env

            # Check stopping criterion
            zero_crossings = self.count_zero_crossings(new_h)
            extrema_count = len(maxima_numpy) + len(minima_numpy)
            
            if abs(zero_crossings - extrema_count) <= 1:
                return new_h
            
            h = new_h.clone()

        return h
    
    def decompose(self, signal: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Performs the full Empirical Mode Decomposition on a signal.

        Args:
            signal (numpy.ndarray): Input signal to be decomposed.

        Returns:
            tuple: A list of IMFs and the residual signal.
        """
        signal = signal.copy()
        signal_tensor = torch.from_numpy(signal).to(self.device)
        imfs = []
        residual = signal_tensor.clone()

        for _ in range(self.max_imfs):
            # Extract one IMF
            imf = self.sift(residual)

            # Update the residual
            residual -= imf

            # Add the IMF to the list
            imf_numpy = imf.cpu().numpy().copy()
            imfs.append(imf_numpy)

            # Check if the residual is effectively zero
            if torch.all(torch.abs(residual) < self.tol):
                break

        return imfs, residual.cpu().numpy().copy()
