import sys 
sys.path.append('E:/Deep learning assets (XXXXXXXX)/Classification trainer')
sys.path.append('E:/Amrita/Subjects/Sem 5/BMSP paper work/Code base/Trainer')

import torch
from torch import nn
import matplotlib.pyplot as plt

from trainer import ModularTrainer
from dataset4 import get_data_loaders

from model2 import CoattentionModel



def main():


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    model = CoattentionModel()


    lmdb_path = "E:/Amrita/Subjects/Sem 5/BMSP paper work/Dataset/Final VMD/VMD.lmdb"
    seed = 42
    train_loader, test_loader = get_data_loaders(lmdb_path=lmdb_path, batch_size=5, num_workers=12, prefetch_factor=2, seed=seed)


    learning_rate = 1e-5
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=False)


    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,  
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        log_path = 'E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Logs/coattention_reconstructed_stft_wave2vec.log',
        num_epochs = 24,
        checkpoint_path = "E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Checkpoints/coattention reconstructed stft wave2vec",
        loss_path = 'E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Graphs/coattention_reconstructed_stft_wave2vec.png',
        verbose=True,
        device=device
    )

    #trainer.train(resume_from="E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Checkpoints/Convxnet stft only/model_epoch_15.pth")
    trainer.train()



if __name__ == '__main__':
    main()
