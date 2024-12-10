from train.object_detection_trainer import ObjectDetectionTrainer


if __name__ == '__main__':
    # Initialize trainer
    trainer = ObjectDetectionTrainer('configs/training_config.yaml')

    # Start training
    trainer.train(epochs=100)

    # Export model
    trainer.export_model('exported_models')
    