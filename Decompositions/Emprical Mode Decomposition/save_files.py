import os
import librosa
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from PIL import Image




def save_modes_and_reconstruct(modes, sample_rate, original_signal, base_filename, output_dir='EMD/Outputs', exclude_modes=None):
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
    
    # Calculate and return reconstruction error
    reconstruction_error = np.mean((original_signal - reconstructed_signal) ** 2)
    
    return reconstruction_error

    
    
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

    
    
    
    
    

