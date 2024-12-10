# train/yolo_trainer.py
from ultralytics import YOLO
import torch
from pathlib import Path
from .base_trainer import BaseTrainer
import yaml
import logging
from typing import Dict, Optional

class YOLONanoTrainer(BaseTrainer):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        self.model = None
        self.setup_model()

    def setup_model(self):
        """Initialize YOLO model"""
        try:
            # Initialize YOLONano model
            if self.config['model'].get('pretrained'):
                # Load pretrained model
                self.model = YOLO('yolov8n.pt')
                self.logger.info("Loaded pretrained YOLOv8-nano model")
            else:
                # Create new model from scratch
                self.model = YOLO('yolov8n.yaml')
                self.logger.info("Created new YOLOv8-nano model")

            # Update model configuration
            self.model.overrides['imgsz'] = self.config['model']['input_size']
            self.model.overrides['batch'] = self.config['training']['batch_size']
            
        except Exception as e:
            self.logger.error(f"Error setting up model: {str(e)}")
            raise

    def train(self):
        """Train the model using Ultralytics trainer"""
        try:
            # Prepare training arguments
            train_args = {
                'data': self.config['data']['yaml_path'],
                'epochs': self.config['training']['epochs'],
                'imgsz': self.config['model']['input_size'],
                'batch': self.config['training']['batch_size'],
                'device': self.device,
                'workers': self.config['training'].get('num_workers', 8),
                'patience': self.config['training'].get('patience', 50),
                'save_period': self.config['training'].get('save_period', -1),
                'project': self.config['training']['project_name'],
                'name': self.config['training']['run_name'],
                'exist_ok': True,
                'pretrained': self.config['model'].get('pretrained', True),
                'optimizer': self.config['training'].get('optimizer', 'auto'),
                'verbose': True,
                'seed': 42
            }

            # Add optional configurations if specified
            if 'learning_rate' in self.config['training']:
                train_args['lr0'] = self.config['training']['learning_rate']

            if 'weight_decay' in self.config['training']:
                train_args['weight_decay'] = self.config['training']['weight_decay']

            # Start training
            self.logger.info("Starting training...")
            results = self.model.train(**train_args)
            
            # Log final metrics
            self.logger.info("Training completed. Final metrics:")
            self.logger.info(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
            self.logger.info(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

            return results

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def export_model(self, format: str = 'torchscript'):
        """Export the trained model"""
        try:
            export_path = Path(self.config['export']['output_dir'])
            export_path.mkdir(parents=True, exist_ok=True)

            # Export model
            self.model.export(
                format=format,
                imgsz=self.config['model']['input_size'],
                batch=1,
                device=self.device,
                simplify=True
            )

            self.logger.info(f"Model exported successfully to {export_path}")

        except Exception as e:
            self.logger.error(f"Error exporting model: {str(e)}")
            raise

    def validate(self, data_path: Optional[str] = None):
        """Validate the model"""
        try:
            val_args = {
                'data': data_path or self.config['data']['yaml_path'],
                'batch': self.config['training']['batch_size'],
                'imgsz': self.config['model']['input_size'],
                'device': self.device,
                'verbose': True
            }

            # Run validation
            results = self.model.val(**val_args)
            
            return results

        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            raise