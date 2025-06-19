import os
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import librosa
from TorchEMD import EmpricalModeDecomposition
from save_files import save_modes_and_reconstruct
from scipy.signal import wiener
from PyEMD import EMD


def process_single_batch(csv_path, audio_dir, start_idx, end_idx, out_dir, device: torch.device = 'cpu'):
    """Process a single batch of files"""

    # Read the CSV file
    df = pd.read_csv(csv_path)
    df_batch = df.iloc[start_idx:end_idx+1]
    
    # Initialize EMD
    emd = EmpricalModeDecomposition(
        max_imfs=10,
        max_iter=128,
        tol=1e-8,
        device=device
    )

    # Initialize EMD\
    emd = EMD(MAX_ITERATION=128, DTYPE=np.float32)
    
    for _, row in tqdm(df_batch.iterrows(), total=len(df_batch)):
        
        relative_path = row['AudioPath'].strip()
        if relative_path.startswith('data/audios/'):
            relative_path = relative_path.replace('data/audios/', '')
        
        file_path = os.path.join(audio_dir, relative_path)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            data, sr = librosa.load(file_path)
            
            if len(data.shape) == 2:
                data = data[:, 0]
            data = data / np.max(np.abs(data))
            data = wiener(data)
            
            # Decompose signal
            imfs = emd.emd(data, max_imf=9)
            
            # Save the decomposed modes and reconstructed signal
            save_modes_and_reconstruct(imfs, sr, data, base_filename, out_dir)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":

    # Device agnostic code
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = 'cpu'

    csv_path = "E:/Amrita/Subjects/Sem 5/BMSP paper work/Dataset/Final EMD/final_metadata.csv"
    audio_dir = "E:/Amrita/Subjects/Sem 5/BMSP paper work/Dataset/neurovoz_v3/data/audios"
    start_idx = 0
    end_idx = 2701
    out_dir = "E:/Amrita/Subjects/Sem 5/BMSP paper work/Dataset/Final EMD/Unprocessed"
    
    process_single_batch(csv_path, audio_dir, start_idx, end_idx, out_dir, device)