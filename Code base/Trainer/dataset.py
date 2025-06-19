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

        imf = data['IMFs']
        spectrogram = data['Spectrograms']
        label = torch.tensor(data['Label'], dtype=torch.float32)

        zero_padding = torch.zeros(10 - spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2])
        img_mask = torch.zeros(640, dtype=torch.bool)
        img_mask[spectrogram.shape[0]*64:] = True
        spectrogram = torch.cat((spectrogram, zero_padding), dim=0)

        imfs = []
        audio_mask = torch.zeros(1440, dtype=torch.bool)
        for i in range(imf.shape[0]):
            pad_size = (1024 - (imf[i].shape[0] % 1024)) % 1024
            temp = F.pad(imf[i], (0,pad_size), mode='constant', value=0)
            temp = temp.view(-1, 1024)
            if temp.shape[0] > 144:
                print(f'\nERROR : Imf has length {temp.shape[0]*1024}')
                imfs = [torch.ones(144, 1024)] * 10
                audio_mask = torch.ones(1440, dtype=torch.bool)
                break
            zero_padding = torch.zeros(144-temp.shape[0], 1024)
            #audio_mask[(i*60)+int(torch.ceil(torch.tensor(temp.shape[0]/8))):((i+1)*60)] = 1
            audio_mask[(i*144)+temp.shape[0]:(i+1)*144] = True
            new_imf = torch.cat([temp, zero_padding], dim=0)
            imfs.append(new_imf)

        imfs = torch.cat(imfs, dim=0)
        zero_padding = torch.zeros(1440-imfs.shape[0], 1024)
        audio_mask[imfs.shape[0]:] = True
        imfs = torch.cat([imfs, zero_padding], dim=0)

        return {
            'IMF' : imfs,
            'Spectrogram' : spectrogram,
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