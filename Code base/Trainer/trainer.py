import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, Union, List
from logger import ContinuousLogger
from tqdm import tqdm
import os


class ModularTrainer:
    
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 val_loader: Optional[DataLoader] = None,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[torch.device] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer with history management.
        """
        # Initialize continuous logger
        self.logger = ContinuousLogger(
            log_dir=config.get('log_dir', './logs') if config else './logs',
            log_file=config.get('log_file', 'training.log') if config else 'training.log'
        )

        # Device configuration
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Model and data
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
 
        # Loss and optimization
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(self.model.parameters())
        self.scheduler = scheduler

        # Configuration
        self.config = config or {}
        self.epochs = self.config.get('epochs', 10)
        self.save_dir = self.config.get('save_dir', './checkpoints')
        self.verbose = self.config.get('verbose', True)
        
        # Training state tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        
        # History tracking
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for a single epoch.
        
        Returns:
            Dict of training metrics
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(enumerate(self.train_loader), 
                             desc=f"Training (Epoch {self.current_epoch}/{self.epochs})", 
                             disable=not self.verbose,
                             total=len(self.train_loader))
        
        for i, batch in progress_bar:

            imfs = batch['IMF'].to(self.device)
            spectrograms = batch['Spectrogram'].to(self.device)
            labels = batch['Label'].to(self.device).unsqueeze(dim=1)
            # img_mask = batch['Image Mask'].to(self.device)
            # audio_mask = batch['Audio Mask'].to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            # outputs = self.model(image=spectrograms, audio=imfs, img_mask=img_mask, audio_mask=audio_mask)
            outputs = self.model(image=spectrograms, audio=imfs)
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'batch loss': loss.item(), 
                'train loss': total_loss/(i+1)
            })
        
        return {
            'train_loss': total_loss / len(self.train_loader),
            'global_step': self.global_step
        }

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dict of validation metrics
        """
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.val_loader), 
                                 desc="Validation", 
                                 disable=not self.verbose,
                                 total=len(self.val_loader))
            
            for i, batch in progress_bar:

                imfs = batch['IMF'].to(self.device)
                spectrograms = batch['Spectrogram'].to(self.device)
                labels = batch['Label'].to(self.device).unsqueeze(dim=1)
                # img_mask = batch['Image Mask'].to(self.device)
                # audio_mask = batch['Audio Mask'].to(self.device)
                
                # outputs = self.model(image=spectrograms, audio=imfs, img_mask=img_mask, audio_mask=audio_mask)
                outputs = self.model(image=spectrograms, audio=imfs)
                
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'batch loss': loss.item(),
                    'test loss': total_loss/(i+1)
                })
        
        val_loss = total_loss / len(self.val_loader)
        val_accuracy = 100 * correct / total

        if self.scheduler:
            self.scheduler.step(val_loss)
        
        if self.verbose:
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Update best metric (assuming lower validation loss is better)
        if val_loss < self.best_metric:
            self.best_metric = val_loss
            self.save_checkpoint(is_best=True)
        
        return {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }


    def train(self, 
              resume_from: Optional[Union[str, Dict[str, Any]]] = None,
              callback: Optional[Callable[[Dict[str, float]], None]] = None) -> Dict[str, List[float]]:
        """
        Main training loop with history resumption.
        
        Args:
            resume_from (str or dict, optional): Checkpoint to resume from
            callback (callable, optional): Function to call after each epoch
        
        Returns:
            Dictionary of training history
        """
        # Attempt to resume from checkpoint if provided
        if resume_from:
            # Load checkpoint and potentially restore previous history
            self.load_checkpoint(resume_from)
            # Log detailed resume information
            self.logger.log_training_resume(
                epoch=self.current_epoch, 
                global_step=self.global_step, 
                total_epochs=self.epochs
            )
        else:
            self.logger.info(f"Starting new training for {self.epochs} epochs")
        
        # Logging and tracking
        print(f"Starting training from epoch {self.current_epoch + 1} to {self.epochs}")
        print(f"Previous history length: {len(self.history['train_loss'])}")
        
        # Continue training from the last completed epoch
        for epoch in range(self.current_epoch + 1, self.epochs + 1):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Store training loss
            self.history['train_loss'].append(train_metrics['train_loss'])
            
            # Validate if validation loader exists
            if self.val_loader:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics.get('val_loss', 0))
                self.history['val_accuracy'].append(val_metrics.get('val_accuracy', 0))
            
            # Optional callback
            if callback:
                callback({**train_metrics, **val_metrics})
            
            # Save checkpoint
            self.save_checkpoint()
        
        return self.history

    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint including training history.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,  # Save entire training history
            'best_metric': self.best_metric
        }
        
        # Define checkpoint filename
        if is_best:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            checkpoint_path = os.path.join(
                self.save_dir, 
                f'model_epoch_{self.current_epoch}.pth'
            )
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        if self.verbose:
            save_type = "Best model" if is_best else "Checkpoint"
            self.logger.info(f"{save_type} saved to {checkpoint_path}")

    def load_checkpoint(self, 
                        checkpoint: Optional[Union[str, Dict[str, Any]]] = None,
                        resume_from_best: bool = False):
        """
        Load model checkpoint.
        
        Args:
            checkpoint (str or dict, optional): Checkpoint to load
            resume_from_best (bool): Load the best model checkpoint
        
        Returns:
            int: Epoch number of the loaded checkpoint
        """
        
        if resume_from_best:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Load checkpoint from file or dictionary
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)
        elif not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a file path or a state dictionary")
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        # Restore training history
        # Use get() to handle cases where history might not be in checkpoint
        loaded_history = checkpoint.get('history', {})
        
        # Merge or replace history
        for key in self.history:
            # Extend existing history or use loaded history
            self.history[key] = loaded_history.get(key, self.history[key])
        
        print(f"Resumed training from epoch {self.current_epoch}")
        print(f"Restored history length: {len(self.history['train_loss'])}")
        
        return self.current_epoch