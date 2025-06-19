import sys 
sys.path.append('E:/Deep learning assets (XXXXXXXX)/Classification trainer')
sys.path.append('E:/Amrita/Subjects/Sem 5/BMSP paper work/Code base/Trainer')

import torch
from torch import nn
import matplotlib.pyplot as plt

from trainer import ModularTrainer
from dataset2 import get_data_loaders

from torchvision import models


def main():


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
    model = models.convnext_small(weights=weights).to(device)

    new_first_layer = nn.Conv2d(
        in_channels=16,
        out_channels=96,   
        kernel_size=4,
        stride=4,
        padding=0
    ) 

    with torch.no_grad():
        new_first_layer.weight[:, :3] = model.features[0][0].weight.clone()
        new_first_layer.weight[:, 3:] = torch.mean(model.features[0][0].weight, dim=1, keepdim=True).repeat(1, 13, 1, 1)
        
    model.features[0][0] = new_first_layer
    model.classifier[2] = nn.Linear(in_features=768, out_features=1, bias=True)

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
        log_path = 'E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Logs/convxnet_mel_stft_both.log',
        num_epochs = 16,
        checkpoint_path = "E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Checkpoints/Convxnet mel stft both",
        loss_path = 'E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Graphs/convxnet_mel_stft_both.png',
        verbose=True,
        device=device
    )

    #trainer.train(resume_from="E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Checkpoints/Convxnet stft only/model_epoch_15.pth")
    trainer.train()


if __name__ == '__main__':
    main()
