# batch_launcher.py
import subprocess
import pandas as pd
import sys
import time

def launch_batch_processor(csv_path, audio_dir, batch_size=100):
    """Launch separate Python processes for each batch with clean memory"""
    # Read the CSV file to get total number of files
    df = pd.read_csv(csv_path)
    total_files = len(df)
    num_batches = (total_files + batch_size - 1) // batch_size
    
    print(f"Processing {total_files} files in {num_batches} batches")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_files)
        
        print(f"\nLaunching batch {batch_idx + 1}/{num_batches}")
        print(f"Processing files {start_idx} to {end_idx}")
        
        # Launch a new Python process for this batch
        cmd = [
            sys.executable,
            "VMD/process_batch.py",
            "--csv_path", csv_path,
            "--audio_dir", audio_dir,
            "--start_idx", str(start_idx),
            "--end_idx", str(end_idx)
        ]
        
        try:
            # Run the batch processor and wait for it to complete
            subprocess.run(cmd, check=True)
            print(f"Completed batch {batch_idx + 1}/{num_batches}")
        except subprocess.CalledProcessError as e:
            print(f"Error in batch {batch_idx + 1}: {e}")
            
        # Wait a few seconds to ensure GPU memory is fully released
        time.sleep(5)

if __name__ == "__main__":
    csv_path = r"./neurovoz_v3/data/audio_features/audio_features.csv"
    audio_dir = r"./neurovoz_v3/data/audios"
    launch_batch_processor(csv_path, audio_dir)