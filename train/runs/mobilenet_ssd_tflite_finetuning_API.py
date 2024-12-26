"""
MobileNet-SSD Fine-tuning Script
Downloads pre-trained model and config, then modifies only the necessary paths
"""

import os
import tensorflow as tf
import wget
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def download_model_and_config():
    """Download pre-trained model and config"""
    base_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/"
    model_name = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Download model
    if not os.path.exists(f"models/{model_name}.tar.gz"):
        print("Downloading model...")
        wget.download(f"{base_url}{model_name}.tar.gz", f"models/{model_name}.tar.gz")
        
        # Extract model
        import tarfile
        with tarfile.open(f"models/{model_name}.tar.gz") as tar:
            tar.extractall("models")
    
    return f"models/{model_name}/pipeline.config"

def modify_config(config_path):
    """Modify config file with new data paths"""
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    
    # Modify data paths
    pipeline_config.train_input_reader.label_map_path = "../../datasets/tfrecord_data_duck/label_map.pbtxt"
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = ["../../datasets/tfrecord_data_duck/train.record"]
    
    pipeline_config.eval_input_reader[0].label_map_path = "../../datasets/tfrecord_data_duck/label_map.pbtxt"
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = ["../../datasets/tfrecord_data_duck/val.record"]
    
    # Modify training config
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = "models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_config.num_steps = 50000
    
    # Modify model config for single class
    pipeline_config.model.ssd.num_classes = 1  # Just for duck class
    
    # Save modified config
    config_text = text_format.MessageToString(pipeline_config)
    with open('pipeline.config', 'w') as f:
        f.write(config_text)
    
    return pipeline_config

def main():
    # Download model and get config
    base_config_path = download_model_and_config()
    print(f"Downloaded config to: {base_config_path}")
    
    # Modify config
    pipeline_config = modify_config(base_config_path)
    print("Modified config saved to: pipeline.config")
    
    # Import training module
    from object_detection import model_main_tf2
    
    # Set training parameters
    flags = tf.compat.v1.flags
    flags.DEFINE_string('model_dir', 'training/', 'Path to output model directory')
    flags.DEFINE_string('pipeline_config_path', 'pipeline.config', 'Path to pipeline config file')
    flags.DEFINE_integer('num_train_steps', 50000, 'Number of train steps')
    flags.DEFINE_bool('eval_training_data', False, 'If training data should be evaluated for this job.')
    flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of every n eval input examples')
    
    # Start training
    tf.compat.v1.app.run(model_main_tf2.main)

if __name__ == '__main__':
    main()