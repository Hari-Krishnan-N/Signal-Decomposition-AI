import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms

import lmdb
import pickle

import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, lmdb_path, channels=8):
        self.lmdb_path = lmdb_path
        self.channels = channels
        self.keys = self._get_keys()

    def _get_keys(self):
        # Open the LMDB environment to fetch all keys
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                keys = [key for key, _ in txn.cursor()]  # Collect keys
        return keys

    def __len__(self):
        return len(self.keys)  # Return the length of keys

    def __getitem__(self, idx):
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                data_bytes = txn.get(self.keys[idx])
                data = pickle.loads(data_bytes)  # Deserialize data

        spectrogram = data['spectrogram']  # 8-channel spectrogram
        label = data['label']  # Label

        if self.channels == 1:
            spectrogram = torch.mean(spectrogram, dim=0, keepdim=True)  # Average 8 channels to 1

        return spectrogram, torch.tensor(label, dtype=torch.long)


def get_data_loaders(train_path, val_path, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    
    dataset = SpectrogramDataset(lmdb_path='output.lmdb')

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)




def get_dataset(channels=8, seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    dataset = SpectrogramDataset(lmdb_path='output.lmdb', channels=channels)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


# def get_dataloader(base_path='C:/Users/Arun/pytorch/datasets/final_hand', 
#                         datasets=None,
#                         batch_size=32,
#                         input_size=(512, 512),
#                         num_workers=4,
#                         preload=False,
#                         prefetch_factor=2,
#                         seed=42,
#                         channels=8):
    
#     train_dataset, val_dataset = get_dataset(base_path, datasets, input_size, preload,seed=seed, channels=channels)

#     dataloader_args = {
#         'batch_size': batch_size,
#         'pin_memory': True,
#     }

#     if num_workers > 0:
#         dataloader_args.update({
#             'num_workers': num_workers,
#             'prefetch_factor': prefetch_factor,
#             'persistent_workers': True,
#             'worker_init_fn': worker_init_fn
#         })

#     train_loader = DataLoader(
#         train_dataset,
#         shuffle=True,
#         **dataloader_args
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         shuffle=False,
#         **dataloader_args
#     )

#     return train_loader, val_loader

import torchaudio
import os
import pandas as pd

class ParkinsonDataset(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None):
        self.audio_metadata = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.audio_metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        audio_name = self.audio_metadata.iloc[idx, 0]
        audio_path = self.audio_metadata.iloc[idx, 1]
        label = self.audio_metadata.iloc[idx, 2]
        
        full_path = os.path.join(self.audio_dir, audio_path)
        waveform, sample_rate = torchaudio.load(full_path, normalize=True)
        if sample_rate != 16000:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resample_transform(waveform)
            sample_rate = 16000
            
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        if self.transform:
            waveform = self.transform(waveform)
            
        # return {
        #     "waveform": waveform,
        #     "sample_rate": sample_rate,
        #     "label": torch.tensor(label, dtype=torch.long)
        # }
        return waveform, torch.tensor(label, dtype=torch.long)
# def collate_fn(batch):
#     # Get max length in batch
#     max_length = max([item['waveform'].shape[1] for item in batch])
    
#     # Pad all waveforms to max length
#     waveforms = []
#     for item in batch:
#         wav = item['waveform']
#         padding_needed = max_length - wav.shape[1]
#         padded_wav = torch.nn.functional.pad(wav, (0, padding_needed))
#         waveforms.append(padded_wav)
    
#     # return {
#     #     "waveform": torch.stack(waveforms),
#     #     "sample_rate": torch.tensor([item['sample_rate'] for item in batch]),
#     #     "label": torch.stack([item['label'] for item in batch])
#     # }
#         # print(torch.stack(waveforms).shape)
#         # print(torch.stack([item['label'] for item in batch]).shape)
#     return torch.stack(waveforms), torch.stack([item['label'] for item in batch])

def collate_fn(batch):
    waveforms, labels = zip(*batch)
    max_length = max(wav.shape[1] for wav in waveforms)
    
    padded_waveforms = [
        torch.nn.functional.pad(wav, (0, max_length - wav.shape[1]))
        for wav in waveforms
    ]
    
    waveforms_tensor = torch.stack(padded_waveforms)
    waveforms_tensor = waveforms_tensor.squeeze(2)  # Remove extra dimension if present
    
    return torch.stack(padded_waveforms), torch.stack(labels)

def get_dataloader(base_path=r"../neurovoz_v3",
                   data_type='spectrogram',  # 'spectrogram' or 'audio'
                   csv_path=r"../VMD/audio_metadata1.csv",  # Required for audio data
                   batch_size=32,
                   num_workers=4,
                   prefetch_factor=2,
                   channels=8,
                   seed=42):
    
    dataloader_args = {
        'batch_size': batch_size,
        'pin_memory': True,
    }

    if num_workers > 0:
        dataloader_args.update({
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': True,
            'worker_init_fn': worker_init_fn
        })

    if data_type == 'spectrogram':
        train_dataset, val_dataset = get_dataset(channels=channels, seed=seed)
    else:  # audio
        torch.manual_seed(seed)  # Set the seed for audio dataset split
        dataset = ParkinsonDataset(csv_path, base_path)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        dataloader_args['collate_fn'] = collate_fn

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_args
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_args
    )

    return train_loader, val_loader
