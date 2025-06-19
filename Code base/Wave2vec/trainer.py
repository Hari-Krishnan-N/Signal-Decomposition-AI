import itertools
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
                 device: Optional[torch.device] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer with binary cross-entropy loss.
        """
        # Initialize continuous logger
        self.logger = ContinuousLogger(
            log_dir=config.get('log_dir', './logs') if config else './logs'
        )

        # Device configuration
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Model and data
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss and optimization
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

        # Configuration
        self.config = config or {}
        self.epochs = self.config.get('epochs', 10)
        self.save_dir = self.config.get('save_dir', './checkpoints')
        self.verbose = self.config.get('verbose', True)
        
        # Training state tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.batch_log_frequency = 10
        
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        
        # History tracking
        self.history: Dict[str, List[float]] = {
            'train_loss_batch': [],
            'train_acc_batch': [],
            'val_loss_batch': [],
            'val_acc_batch': [],
            'train_loss_epoch': [],
            'train_acc_epoch': [],
            'val_loss_epoch': [],
            'val_acc_epoch': []
        }

    def compute_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute binary classification accuracy.
        """
        with torch.no_grad():
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            return (correct / total) * 100

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.epochs}")
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(self.device), labels.float().to(self.device)

            if inputs.dim() == 3:
                inputs = inputs.squeeze(1)  # Ensure inputs have the correct shape
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs).logits.squeeze()  # Ensure outputs are squeezed
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy for binary classification
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            
            # Update running metrics
            running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
            running_correct += correct
            running_total += total
            running_acc = (running_correct / running_total) * 100
            
            progress_bar.set_postfix({'loss': f'{running_loss:.4f}', 'acc': f'{running_acc:.2f}%'})
            
            if (batch_idx + 1) % self.batch_log_frequency == 0:
                self.history['train_loss_batch'].append(running_loss)
                self.history['train_acc_batch'].append(running_acc)
                
                if self.val_loader:
                    val_metrics = self.validate_step()
                    self.history['val_loss_batch'].append(val_metrics['val_loss'])
                    self.history['val_acc_batch'].append(val_metrics['val_accuracy'])
        
        epoch_acc = (running_correct / running_total) * 100
        epoch_loss = running_loss
        
        return {'train_loss': epoch_loss, 'train_accuracy': epoch_acc, 'global_step': self.global_step}

    def validate_step(self) -> Dict[str, float]:
        """
        Perform a quick validation step for binary classification.
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        batch_count = 0
        
        with torch.no_grad():
            for inputs, labels in itertools.islice(self.val_loader, self.batch_log_frequency):
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                
                if inputs.dim() == 3:
                    inputs = inputs.squeeze(1)
                
                outputs = self.model(inputs).logits.squeeze()
                loss = self.criterion(outputs, labels)
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()
                batch_count += 1
        
        return {'val_loss': val_loss / batch_count, 'val_accuracy': 100 * val_correct / val_total}

    def validate(self) -> Dict[str, float]:
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                
                if inputs.dim() == 3:
                    inputs = inputs.squeeze(1)
                
                outputs = self.model(inputs).logits.squeeze()
                loss = self.criterion(outputs, labels)
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                batch_correct = (predicted == labels).sum().item()
                batch_total = labels.size(0)
                
                total_loss += loss.item()
                total += batch_total
                correct += batch_correct
            
                progress_bar.set_postfix({'loss': f'{total_loss / (batch_idx + 1):.4f}', 'acc': f'{(correct / total) * 100:.2f}%'})
        
        val_loss = total_loss / len(self.val_loader)
        val_accuracy = 100 * correct / total
        
        self.history['val_loss_epoch'].append(val_loss)
        self.history['val_acc_epoch'].append(val_accuracy)
        
        if val_loss < self.best_metric:
            self.best_metric = val_loss
            self.save_checkpoint(is_best=True)
        
        self.scheduler.step(val_loss)
        
        return {'val_loss': val_loss, 'val_accuracy': val_accuracy}

    # The train, save_checkpoint, and load_checkpoint methods remain unchanged
    def train(self, 
              resume_from: Optional[Union[str, Dict[str, Any]]] = None,
              callback: Optional[Callable[[Dict[str, float]], None]] = None) -> Dict[str, List[float]]:
        if resume_from:
            self.load_checkpoint(resume_from)
        
        for epoch in range(self.current_epoch + 1, self.epochs + 1):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            self.history['train_loss_epoch'].append(train_metrics['train_loss'])
            self.history['train_acc_epoch'].append(train_metrics['train_accuracy'])
            
            if self.val_loader:
                val_metrics = self.validate()
            
            if callback:
                callback({**train_metrics, **(val_metrics or {})})
            
            self.save_checkpoint()
        
        return self.history
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_metric': self.best_metric
        }
        
        import time
        max_retries = 3
        
        def safe_save(save_path: str):
            temp_path = save_path + '.tmp'
            for attempt in range(max_retries):
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    torch.save(checkpoint, temp_path)
                    os.replace(temp_path, save_path)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait a second before retrying
                    else:
                        raise
                except Exception as e:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise e
        
        # Save latest model
        latest_path = os.path.join(self.save_dir, 'latest_model.pth')
        safe_save(latest_path)
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            safe_save(best_path)
        
        if self.verbose:
            save_type = "Best model" if is_best else "Latest model"
            self.logger.info(f"{save_type} saved")

    def load_checkpoint(self, 
                       checkpoint: Optional[Union[str, Dict[str, Any]]] = None,
                       load_best: bool = False):
        if load_best:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        elif isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device)
        elif not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a file path or dict")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        return self.current_epoch