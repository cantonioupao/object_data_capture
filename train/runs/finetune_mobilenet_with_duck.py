from train.object_detection_trainer import ObjectDetectionTrainer

# Initialize trainer with duck-specific config
trainer = ObjectDetectionTrainer('configs/duck_training_tf_config.yaml')

# The existing train() method will handle:
# - Loading the small duck dataset
# - Applying appropriate augmentations
# - Fine-tuning with the specified learning rate
# - Saving checkpoints
# - Monitoring metrics
trainer.train(epochs=20)

# Export the fine-tuned model
trainer.export_model('models/mobilenet_ssd_tflite_duck/duck_model.tflite')