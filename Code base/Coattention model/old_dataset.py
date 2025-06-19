import os
import torch 
from torchvision import transforms
from torchaudio import load
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd



img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize(size=(512, 512)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])



class DecompositionData(Dataset):

    def __init__(self, data_dir:str, metadata_dir:str, img_transform:transforms):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_dir)
        self.img_transform = img_transform


    def __len__(self):
        return self.metadata.shape[0]


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        file_name = self.metadata.iloc[idx, 0]
        label = self.metadata.iloc[idx, 2]
        img_dir = os.path.join(self.data_dir, file_name, 'Spectogram')
        audio_dir = os.path.join(self.data_dir, file_name, 'IMF')

        images = []
        audios = []

        for i in os.listdir(img_dir):
            img = Image.open(os.path.join(img_dir, i))
            if self.img_transform:
                img = self.img_transform(img)
            images.append(img)
        images = torch.cat(images, dim=0)

        for i in os.listdir(audio_dir):
            audio = load(os.path.join(audio_dir, i))
            audios.append(audio)
        audios = torch.cat(audios, dim=0)

        return {
            'Images' : images,
            'Audios' : audios,
            'Label' : label
        }