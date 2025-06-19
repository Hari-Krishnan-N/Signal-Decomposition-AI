from transformers import Wav2Vec2ForSequenceClassification
import torch
import matplotlib.pyplot as plt
from trainer import ModularTrainer
from dataset import get_dataloader

def main():
    # Initialize wav2vec model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=1,
        output_hidden_states=True
    )
    
    # Freeze feature extractor layers (optional)
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False

    train_loader, val_loader = get_dataloader(
        data_type='audio',
        batch_size=8,  # Smaller batch size due to model size
        num_workers=8, 
        prefetch_factor=2, 
    )

    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'epochs': 100,
            'save_dir': './wav2vec_checkpoints',
            'log_dir': './wav2vec_logs',
            'learning_rate': 1e-4
        }
    )

    history = trainer.train()

    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Wav2Vec Training Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()