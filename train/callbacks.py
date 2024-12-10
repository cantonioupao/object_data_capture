import tensorflow as tf
from pathlib import Path
from typing import Dict
import json
import time

class TrainingCallbacks:
    """Callbacks for training monitoring and control"""
    
    def __init__(self, callback_config: Dict):
        self.config = callback_config
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_start = time.time()
        
        # Initialize TensorBoard if configured
        if 'tensorboard' in callback_config:
            self.tensorboard = tf.summary.create_file_writer(
                callback_config['tensorboard']['log_dir']
            )
        else:
            self.tensorboard = None
            
        # Setup checkpoint directory if needed
        if 'model_checkpoint' in callback_config:
            self.checkpoint_dir = Path(callback_config['model_checkpoint'].get(
                'dir', 'checkpoints'
            ))
            self.checkpoint_dir.mkdir(exist_ok=True)
            
    def on_batch_end(self, metrics: Dict):
        """Called at the end of each training batch"""
        if self.tensorboard:
            with self.tensorboard.as_default():
                for name, value in metrics.items():
                    tf.summary.scalar(f'batch_{name}', value, step=self.training_step)
                    
        self.training_step += 1
        
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Called at the end of each epoch"""
        # Log metrics
        self._log_metrics(epoch, metrics)
        
        # Check for model saving
        if self._should_save_model(metrics):
            self._save_checkpoint(epoch, metrics)
            
        # Check for early stopping
        if self._should_stop_training(metrics):
            return True
            
        return False
        
    def _log_metrics(self, epoch: int, metrics: Dict):
        """Log metrics to TensorBoard and console"""
        # Console logging
        print(f"\nEpoch {epoch + 1} Results:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
            
        # TensorBoard logging
        if self.tensorboard:
            with self.tensorboard.as_default():
                for name, value in metrics.items():
                    tf.summary.scalar(f'epoch_{name}', value, step=epoch)
                    
    def _should_save_model(self, metrics: Dict) -> bool:
        """Determine if model should be saved"""
        if 'model_checkpoint' not in self.config:
            return False
            
        monitor = self.config['model_checkpoint'].get('monitor', 'val_loss')
        current_value = metrics.get(monitor)
        
        if current_value is None:
            return False
            
        if current_value < self.best_loss:
            self.best_loss = current_value
            return True
            
        return False
        
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.h5"
        
        # Save metrics along with checkpoint
        metrics_path = checkpoint_path.with_suffix('.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'epoch': epoch + 1,
                'metrics': metrics,
                'timestamp': time.time()
            }, f, indent=2)
            
    def _should_stop_training(self, metrics: Dict) -> bool:
        """Check if training should be stopped"""
        if 'early_stopping' not in self.config:
            return False
            
        monitor = self.config['early_stopping'].get('monitor', 'val_loss')
        patience = self.config['early_stopping'].get('patience', 10)
        current_value = metrics.get(monitor)
        
        if current_value is None:
            return False
            
        if current_value >= self.best_loss:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                return True
        else:
            self.patience_counter = 0
            
        return False