import numpy as np
import tensorflow as tf
from core.base import ObjectDetector

class TFLiteDetector(ObjectDetector):
    def __init__(self, model_path='models/mobilenet_v1.tflite', 
                 model_labels_path='models/labelmap.txt',
                 confidence_threshold=0.3):
        # Store configuration parameters
        self.confidence_threshold = confidence_threshold
        
        try:
            # Initialize TFLite interpreter with error handling
            self.interpreter = tf.lite.Interpreter(model_path)
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
            
            # Load labels with error handling
            self.labels = self._load_labels(model_labels_path)
            
            # Print initialization success information
            print(f"Model initialized successfully:")
            print(f"Input shape: {self.input_shape}")
            print(f"Number of classes: {len(self.labels)}")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def detect_object(self, frame):
        try:
            # Verify frame is not None and has correct shape
            if frame is None or len(frame.shape) != 3:
                print("Invalid input frame")
                return None
                
            # Preprocess image
            input_data = self._preprocess(frame)
            
            # Set input tensor and run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get detection results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            
            # Find best detection
            max_idx = np.argmax(scores)
            max_score = scores[max_idx]
            
            # Check confidence threshold
            if max_score < self.confidence_threshold:
                print(f"No detections above threshold {self.confidence_threshold}")
                return None
                
            # Convert normalized coordinates to pixel coordinates
            box = boxes[max_idx]
            h, w = frame.shape[:2]
            x = int(box[1] * w)
            y = int(box[0] * h)
            width = int((box[3] - box[1]) * w)
            height = int((box[2] - box[0]) * h)
            
            # Ensure coordinates are within frame bounds
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            width = max(0, min(width, w - x))
            height = max(0, min(height, h - y))
            
            detected_class = self.labels[int(classes[max_idx])]
            print(f"Detected {detected_class} with confidence {max_score:.2f}")
            
            return {
                'bounds': (x, y, width, height),
                'class': detected_class,
                'score': float(max_score)
            }
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return None

    def _preprocess(self, frame):
        try:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
                
            # Resize frame to match model's input shape
            resized = tf.image.resize(frame, (self.input_shape[1], self.input_shape[2]))
            
            # Ensure correct data type
            processed = tf.cast(resized, tf.uint8)
            
            # Add batch dimension
            return np.expand_dims(processed, axis=0)
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def _load_labels(self, model_labels_path):
        try:
            with open(model_labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
                
            if not labels:
                raise ValueError("Empty labels file")
                
            return labels
            
        except Exception as e:
            print(f"Error loading labels: {str(e)}")
            raise