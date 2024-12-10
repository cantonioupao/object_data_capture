from train.yolo_trainer import YOLONanoTrainer

if __name__ == '__main__':
    # Initialize trainer
    trainer = YOLONanoTrainer('configs/yolo_config.yaml')
    
    # Start training
    results = trainer.train()
    
    # Export model
    trainer.export_model(format='torchscript')

