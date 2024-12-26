import tensorflow as tf
from .base_trainer import BaseTrainer
from .data_loader.data_loader import ObjectDetectionDataLoader
from .callbacks import TrainingCallbacks
from .metrics import ObjectDetectionMetrics
from pathlib import Path

class ObjectDetectionTrainer(BaseTrainer):
    """Trainer class specifically for object detection models"""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Initialize components
        self.data_loader = ObjectDetectionDataLoader(
            self.config['data'],
            self.config['augmentation']
        )
        
        self.metrics = ObjectDetectionMetrics()
        self.callbacks = TrainingCallbacks(self.config['callbacks'])
        
        # Build model and optimizer
        self.build_model()
        self.setup_optimizer()
        
    def build_model(self):
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=[224, 224, 3],
            include_top=False,
            weights='imagenet'
        )
        backbone.trainable = False  # Freeze backbone

        features = backbone.output
        x = tf.keras.layers.Conv2D(256, 3, padding='same')(features)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        cls_head = tf.keras.layers.Conv2D(1, 1)(x)
        box_head = tf.keras.layers.Conv2D(4, 1)(x)

        self.model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[cls_head, box_head]
        )
            
    def setup_optimizer(self):
        """Setup optimizer with learning rate schedule"""
        initial_lr = self.config['training']['learning_rate']
        
        # Optional learning rate schedule
        if self.config['training'].get('lr_schedule'):
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_lr,
                decay_steps=1000,
                decay_rate=0.9
            )
            self.optimizer = tf.keras.optimizers.Adam(lr_schedule)
        else:
            self.optimizer = tf.keras.optimizers.Adam(initial_lr)
            
    @tf.function
    def train_step(self, batch):
        """Single training step using gradient tape"""
        images, labels = batch
        
        with tf.GradientTape() as tape:
            # Forward pass
            cls_pred, box_pred = self.model(images, training=True)
            
            # Calculate losses
            cls_loss = self._classification_loss(labels['classes'], cls_pred)
            box_loss = self._box_regression_loss(labels['boxes'], box_pred)
            total_loss = cls_loss + self.config['training']['box_loss_weight'] * box_loss
            
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss
        }
        
    @tf.function
    def validate_step(self, batch):
        """Single validation step"""
        images, labels = batch
        cls_pred, box_pred = self.model(images, training=False)
        
        # Calculate validation metrics
        metrics = self.metrics.update_state(
            labels['classes'],
            labels['boxes'],
            cls_pred,
            box_pred
        )
        
        return metrics
        
    def _classification_loss(self, y_true, y_pred):
        """Binary classification loss per grid cell"""
        return tf.keras.losses.BinaryCrossentropy(
            from_logits=True
        )(y_true, y_pred)

    def _box_regression_loss(self, y_true, y_pred):
        """L1 loss for bounding box regression"""
        # Only compute loss for cells containing objects
        object_mask = tf.cast(y_true[..., 0] > 0, tf.float32)
        return tf.reduce_sum(
            object_mask * tf.abs(y_true - y_pred)
        ) / tf.maximum(tf.reduce_sum(object_mask), 1)
        
    def train(self, epochs: int):
        """Main training loop"""
        for epoch in range(epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            
            # Training phase
            for batch in self.data_loader.train_dataset:
                metrics = self.train_step(batch)
                self.callbacks.on_batch_end(metrics)
                
            # Validation phase
            for batch in self.data_loader.val_dataset:
                metrics = self.validate_step(batch)
                
            # Epoch end processing
            self.callbacks.on_epoch_end(epoch, metrics)
            
            # Save checkpoints
            if (epoch + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch)
                
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.h5"
        self.model.save_weights(str(checkpoint_path))
        
    def export_model(self, export_dir: str):
        """Export model to TFLite format"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        export_path = Path(export_dir) / "model.tflite"
        export_path.parent.mkdir(exist_ok=True)
        export_path.write_bytes(tflite_model)