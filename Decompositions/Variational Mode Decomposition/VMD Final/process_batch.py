# process_batch.py
import pandas as pd
import numpy as np
import librosa
from scipy.signal import wiener
import os
from tqdm import tqdm
import argparse
from main import GPU_VMD, save_modes_and_reconstruct

def process_single_batch(csv_path, audio_dir, start_idx, end_idx):
    """Process a single batch of files"""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    df_batch = df.iloc[start_idx:end_idx]
    
    # Initialize VMD
    vmd = GPU_VMD(
        alpha=200000,
        tau=0,
        n_modes=20,
        dc_component=False,
        init_method=1,
        tolerance=1e-6,
        device='cuda'
    )
    
    for _, row in tqdm(df_batch.iterrows(), total=len(df_batch)):
        relative_path = row['AudioPath'].strip()
        if relative_path.startswith('../data/audios/'):
            relative_path = relative_path.replace('../data/audios/', '')
        
        file_path = os.path.join(audio_dir, relative_path)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            # Read and process audio file
            data, sr = librosa.load(file_path)
            # print(min(data), max(data))

            if len(data.shape) == 2:
                data = data[:, 0]
            data = data / np.max(np.abs(data))
            data = wiener(data)
            
            # Decompose signal
            result = vmd.decompose(data, data.shape[0])
            modes = result.modes
            
            # Pad each mode to the same length of the original signal
            modes_padded = []
            for mode in modes:
                mode_padded = np.pad(mode, (0, len(data) - len(mode)), 'constant')
                modes_padded.append(mode_padded)
            
            # Save the decomposed modes and reconstructed signal
            save_modes_and_reconstruct(modes_padded, sr, data, base_filename)
            
            # print(f"Processed {base_filename}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    
    args = parser.parse_args()
    
    process_single_batch(args.csv_path, args.audio_dir, args.start_idx, args.end_idx)