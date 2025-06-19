import torchvision.models as models
from torch import nn
import torch
import matplotlib.pyplot as plt

from trainer import ModularTrainer
from dataset import get_dataloader

def main():
    # Load the pre-trained ResNet-18 model
    model = models.resnet18()

    # Modify the first convolutional layer to accept 8 channels
    new_first_layer = nn.Conv2d(
        in_channels=8,           # Change to 8 channels
        out_channels=64,         # Same number of output channels as original
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Copy the weights of the first 3 channels and initialize the remaining channels randomly
    with torch.no_grad():
        new_first_layer.weight[:, :3] = model.conv1.weight.clone()
        new_first_layer.weight[:, 3:] = torch.mean(model.conv1.weight, dim=1, keepdim=True).repeat(1, 5, 1, 1)

    # Replace the original first layer with the new one
    model.conv1 = new_first_layer

    # Modify the final fully connected (FC) layer for 2 classes
    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    seed = 42  # Define the seed variable
    train_loader, val_loader = get_dataloader(batch_size=32, num_workers=2, prefetch_factor=2, seed=seed)

    # Assuming train_loader and val_loader are defined elsewhere
    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,  
        val_loader=val_loader,
        config={'epochs': 20,
                'save_dir': './model_checkpoints',
                'log_dir': './training_logs'  # Specify log directory
        }
    )

    # First training run
    history = trainer.train()

    # Plotting the training loss
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# Ensure safe multiprocessing on Windows
if __name__ == '__main__':
    main()
