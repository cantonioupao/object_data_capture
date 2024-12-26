import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable

class ObjectDetectionDataLoader:
    """Data loader for object detection datasets with preprocessing capabilities"""
    
    def __init__(self, data_config: Dict, augmentation_config: Dict):
        self.config = data_config
        self.aug_config = augmentation_config
        self.batch_size = data_config['batch_size']
        
        # Get model input configuration
        self.target_size = tuple(data_config.get('target_size', (224, 224)))
        self.preprocessing_function = self._get_preprocessing_function(
            data_config.get('preprocessing', 'mobilenet_v2')
        )
        
        # Initialize datasets
        self.train_dataset = self._create_dataset(data_config['train_path'], is_training=True)
        self.val_dataset = self._create_dataset(data_config['val_path'], is_training=False)
        
    def _get_preprocessing_function(self, model_name: str) -> Callable:
        """Get preprocessing function based on model type"""
        preprocessing_functions = {
            'mobilenet_v2': tf.keras.applications.mobilenet_v2.preprocess_input,
            'resnet50': tf.keras.applications.resnet50.preprocess_input,
            'none': lambda x: x / 255.0  # Simple normalization
        }
        return preprocessing_functions.get(model_name, preprocessing_functions['none'])
        
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
        
        # Apply preprocessing
        dataset = dataset.map(
            self._preprocess_data,
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
        
    def _preprocess_data(self, image: tf.Tensor, labels: Dict) -> Tuple[tf.Tensor, Dict]:
        """Apply preprocessing to image and adjust bounding boxes"""
        # Get original image dimensions
        original_height = tf.cast(tf.shape(image)[0], tf.float32)
        original_width = tf.cast(tf.shape(image)[1], tf.float32)
        
        # Resize image
        image = tf.image.resize(image, self.target_size)
        
        # Apply model-specific preprocessing
        image = self.preprocessing_function(image)
        
        # Adjust bounding box coordinates for new image size
        if 'boxes' in labels:
            boxes = labels['boxes']
            scale_x = tf.cast(self.target_size[1], tf.float32) / original_width
            scale_y = tf.cast(self.target_size[0], tf.float32) / original_height
            
            boxes = tf.stack([
                boxes[..., 0] * scale_x,
                boxes[..., 1] * scale_y,
                boxes[..., 2] * scale_x,
                boxes[..., 3] * scale_y
            ], axis=-1)
            
            labels['boxes'] = boxes
            
        return image, labels
        
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
        
