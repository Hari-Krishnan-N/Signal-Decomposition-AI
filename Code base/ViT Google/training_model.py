import sys 
sys.path.append('E:/Deep learning assets (XXXXXXXX)/Classification trainer')
sys.path.append('E:/Amrita/Subjects/Sem 5/BMSP paper work/Code base/Trainer')

import torch
from torch import nn
import matplotlib.pyplot as plt

from trainer import ModularTrainer
from dataset2 import get_data_loaders

from transformers import ViTImageProcessor, ViTForImageClassification, AutoConfig


def main():


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'




    #processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    config = AutoConfig.from_pretrained('google/vit-base-patch16-224')
    config.pretrained_model_name_or_path = None
    model = ViTForImageClassification._from_config(config)

    new_first_layer = nn.Conv2d(
        in_channels=16,
        out_channels=768,   
        kernel_size=16,
        stride=16,
        padding=0
    )

    with torch.no_grad():
        new_first_layer.weight[:, :3] = model.vit.embeddings.patch_embeddings.projection.weight.clone()
        new_first_layer.weight[:, 3:] = torch.mean(model.vit.embeddings.patch_embeddings.projection.weight, dim=1, keepdim=True).repeat(1, 13, 1, 1)

    model.vit.embeddings.patch_embeddings.projection = new_first_layer
    model.classifier = nn.Linear(in_features=768, out_features=1, bias=True)

    model.config.num_channels = 16
    model.config.num_labels = 1 
    model.vit.embeddings.patch_embeddings.num_channels = 16




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
        log_path = 'E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Logs/vit_google_mel_stft_no_pretraining.log',
        num_epochs = 24,
        checkpoint_path = "E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Checkpoints/vit google mel stft no pretraining",
        loss_path = 'E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Graphs/vit_google_mel_stft_no_pretraining.png',
        verbose=True,
        device=device
    )

    #trainer.train(resume_from="E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Checkpoints/Convxnet stft only/model_epoch_15.pth")
    trainer.train()


if __name__ == '__main__':
    main()
