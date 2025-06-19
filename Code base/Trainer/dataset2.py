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

        log_spectrogram = data['Log Spectrograms']
        mel_spectrogram = data['Mel Spectrogram']
        label = torch.tensor(data['Label'], dtype=torch.float32).unsqueeze(dim=0)

        return {
            'Log Spectrogram' : log_spectrogram,
            'Mel Spectrogram' : mel_spectrogram,
            'Label' : label,
            #'Image Mask' : img_mask,
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