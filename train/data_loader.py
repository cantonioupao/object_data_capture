# train/data_loader.py
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

class ObjectDetectionDataLoader:
    """Data loader for object detection datasets"""
    
    def __init__(self, data_config: Dict, augmentation_config: Dict):
        self.config = data_config
        self.aug_config = augmentation_config
        self.batch_size = data_config['batch_size']
        
        # Initialize datasets
        self.train_dataset = self._create_dataset(data_config['train_path'], is_training=True)
        self.val_dataset = self._create_dataset(data_config['val_path'], is_training=False)
        
    def _create_dataset(self, data_path: str, is_training: bool) -> tf.data.Dataset:
        """Create tf.data.Dataset from directory"""
        path = Path(data_path)
        
        # Create list of image and annotation files
        image_files = sorted(list(path.glob('images/*.jpg')))
        annotation_files = sorted(list(path.glob('annotations/*.xml')))
        
        # Create dataset from files
        dataset = tf.data.Dataset.from_tensor_slices((
            [str(f) for f in image_files],
            [str(f) for f in annotation_files]
        ))
        
        # Apply transformations
        dataset = dataset.map(
            self._parse_data,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if is_training:
            dataset = dataset.map(
                self._augment,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.shuffle(1000)
            
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    def _parse_data(self, image_path: str, annotation_path: str) -> Tuple[tf.Tensor, Dict]:
        """Parse image and annotations"""
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        # Parse XML annotation
        boxes, classes = self._parse_annotation(annotation_path)
        
        return image, {
            'boxes': boxes,
            'classes': classes
        }
        
    def _parse_annotation(self, annotation_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """Parse XML annotation file"""
        # Implementation depends on your annotation format
        # This is a placeholder that should be implemented based on your XML structure
        return tf.zeros((1, 4)), tf.zeros((1,), dtype=tf.int32)
        
    def _augment(self, image: tf.Tensor, labels: Dict) -> Tuple[tf.Tensor, Dict]:
        """Apply augmentations to image and boxes"""
        if self.aug_config['random_flip']:
            image, labels['boxes'] = self._random_flip(image, labels['boxes'])
            
        if self.aug_config['random_brightness']:
            image = tf.image.random_brightness(
                image,
                self.aug_config['random_brightness']
            )
            
        if self.aug_config['random_contrast']:
            image = tf.image.random_contrast(
                image,
                1 - self.aug_config['random_contrast'],
                1 + self.aug_config['random_contrast']
            )
            
        return image, labels
        
    def _random_flip(self, image: tf.Tensor, boxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Randomly flip image and adjust boxes"""
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
            boxes = tf.stack([
                boxes[..., 0],
                1 - boxes[..., 3],
                boxes[..., 2],
                1 - boxes[..., 1]
            ], axis=-1)
            
        return image, boxes
