import os
import cv2
import torch
import torchaudio
from torchvision import transforms
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import lmdb
import pickle
import pandas as pd
from tqdm import tqdm



img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize(size=(512, 512)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])



def load_and_concatenate_data(data_dir:str, file_name:str):

    images = []
    audios = []

    img_dir = os.path.join(data_dir, 'Mel Spectrogram', file_name)
    for i in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, i), cv2.IMREAD_GRAYSCALE)
        img = img_transform(img)
        images.append(img)
    images = torch.cat(images, dim=0)

    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    # model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    audio_dir = os.path.join(data_dir, 'IMFs', file_name)
    for i in os.listdir(audio_dir):
        audio, sr = torchaudio.load(os.path.join(audio_dir, i))
        # resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        # audio = resampler(audio).squeeze(0)
        # audio = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        # audio = model(audio)
        # hidden_states = audio.last_hidden_state
        # audios.append(hidden_states)
        audios.append(audio)
    audios = torch.cat(audios, dim=0)
    
    return images, audios



def save_to_lmdb(lmdb_path, data_dir, metadata_csv):

    metadata = pd.read_csv(metadata_csv)
    map_size = 5e10
    env = lmdb.open(lmdb_path, map_size=int(map_size))

    with env.begin(write=True) as txn:
        
        with tqdm(metadata.iterrows(), total=metadata.shape[0]) as t:

            for idx, row in t:

                filename = row['AudioName']
                label = row['label']

                images, audios = load_and_concatenate_data(data_dir=data_dir, file_name=filename)
                data = {
                    'IMFs' : audios,
                    'Spectrograms': images,
                    'Label': label
                }
                
                data_bytes = pickle.dumps(data)
                txn.put(str(idx).encode(), data_bytes)
    
    print(f"Saved {len(metadata)} samples to LMDB at {lmdb_path}")




def load_from_lmdb(lmdb_path, idx):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        data_bytes = txn.get(str(idx).encode())
        data = pickle.loads(data_bytes)
        return data['Imfs'], data['Spectrograms'], data['Label']


save_to_lmdb(lmdb_path='E:/Amrita/Subjects/Sem 5/BMSP paper work/Dataset/Final VMD/VMD.lmdb',
             data_dir='E:/Amrita/Subjects/Sem 5/BMSP paper work/Dataset/Final VMD',
             metadata_csv='E:/Amrita/Subjects/Sem 5/BMSP paper work/Dataset/Final VMD/final_metadata_2.csv')
