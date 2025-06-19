import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import lmdb
import pickle



class DecompositionDataset(Dataset):
    
    def __init__(self, lmdb_path:str):
        self.lmdb_path = lmdb_path
        self.keys = self._get_keys()

    def _get_keys(self):
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                keys = [key for key, _ in txn.cursor()]
        return keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                data_bytes = txn.get(self.keys[idx])
                data = pickle.loads(data_bytes)

        reconstructed = data['Reconstructed'][0].squeeze(dim=0)
        log_spectrogram = data['Log Spectrograms']
        mel_spectrogram = data['Mel Spectrogram']
        label = torch.tensor(data['Label'], dtype=torch.float32).unsqueeze(dim=0)

        pad_size = (1024 - (reconstructed.shape[0] % 1024)) % 1024
        reconstructed = F.pad(reconstructed, (0,pad_size), mode='constant', value=0)
        reconstructed = reconstructed.view(-1, 1024)
        zero_padding = torch.zeros(144-reconstructed.shape[0], 1024)
        audio_mask = torch.zeros(144, dtype=torch.bool)
        audio_mask[reconstructed.shape[0]:] = True
        reconstructed = torch.cat([reconstructed, zero_padding], dim=0)

        zero_padding = torch.zeros(8 - log_spectrogram.shape[0], log_spectrogram.shape[1], log_spectrogram.shape[2])
        img_mask = torch.zeros(512, dtype=torch.bool)
        img_mask[log_spectrogram.shape[0]*64:] = True
        log_spectrogram = torch.cat((log_spectrogram, zero_padding), dim=0)

        zero_padding = torch.zeros(8 - mel_spectrogram.shape[0], mel_spectrogram.shape[1], mel_spectrogram.shape[2])
        img_mask = torch.zeros(512, dtype=torch.bool)
        img_mask[mel_spectrogram.shape[0]*64:] = True
        mel_spectrogram = torch.cat((mel_spectrogram, zero_padding), dim=0)

        return {
            'Reconstructed' : reconstructed,
            'Log Spectrogram' : log_spectrogram,
            'Mel Spectrogram' : mel_spectrogram,
            'Label' : label,
            'Image Mask' : img_mask,
            'Audio Mask' : audio_mask
        }



def get_data_loaders(lmdb_path, batch_size, num_workers, prefetch_factor, seed=42):
    
    torch.manual_seed(seed)

    dataset = DecompositionDataset(lmdb_path=lmdb_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)

    return train_loader, val_loader