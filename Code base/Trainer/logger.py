
import os
import logging
from logging.handlers import RotatingFileHandler



class ContinuousLogger:
    
    def __init__(self, 
                 log_dir: str = './logs', 
                 log_file: str = 'training.log', 
                 level: int = logging.INFO,
                 max_log_size: int = 10 * 1024 * 1024,  # 10 MB
                 backup_count: int = 5):
        """
        Create a continuous logger with rotation and resume capabilities.
        
        Args:
            log_dir (str): Directory to store log files
            log_file (str): Name of the log file
            level (int): Logging level
            max_log_size (int): Maximum log file size before rotation
            backup_count (int): Number of backup log files to keep
        """
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Full path to log file
        log_path = os.path.join(log_dir, log_file)
        
        # Create logger
        self.logger = logging.getLogger('ModularTrainer')
        self.logger.setLevel(level)
        
        # Clear existing handlers to prevent duplicate logging
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_path, 
            maxBytes=max_log_size, 
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_training_resume(self, 
                             epoch: int, 
                             global_step: int, 
                             total_epochs: int):
        """
        Log detailed information about training resume.
        
        Args:
            epoch (int): Resumed epoch number
            global_step (int): Resumed global training step
            total_epochs (int): Total planned training epochs
        """
        resume_message = (
            f"Training Resumed:\n"
            f"  Current Epoch: {epoch}\n"
            f"  Global Step: {global_step}\n"
            f"  Total Epochs: {total_epochs}\n"
            f"  Remaining Epochs: {total_epochs - epoch}"
        )
        self.info(resume_message)