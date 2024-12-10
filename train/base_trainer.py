from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional
import yaml
import logging
import os

class BaseTrainer(ABC):
    """Abstract base class for all trainers"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = None
        self.optimizer = None
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration from YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
        
    @abstractmethod
    def train_step(self, batch):
        """Single training step"""
        pass
        
    @abstractmethod
    def validate_step(self, batch):
        """Single validation step"""
        pass