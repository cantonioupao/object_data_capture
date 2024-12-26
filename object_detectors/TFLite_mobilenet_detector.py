import numpy as np
import tensorflow as tf
from core.base import ObjectDetector
import cv2
import time


class TFLiteDetector(ObjectDetector):
    def __init__(self, model_path='models/mobilenet_v1.tflite', 
                 model_labels_path='models/labelmap.txt',
                 confidence_threshold=0.3):
        # Store configuration parameters
        self.confidence_threshold = confidence_threshold
        
        # FPS tracking with time window
        self._fps_start_time = time.perf_counter()
        self._fps_counter = 0
        self._fps = 0.0
        self._FPS_UPDATE_INTERVAL = 0.5  # Update FPS every 0.5 seconds
        
        # Separate inference timing
        self.inference_time = 0
        self._last_inference_time = 0
        
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

    def get_fps(self):
        """Get real-time FPS using time window method"""
        current_time = time.perf_counter()
        elapsed = current_time - self._fps_start_time
        
        self._fps_counter += 1
        
        # Update FPS every UPDATE_INTERVAL seconds
        if elapsed >= self._FPS_UPDATE_INTERVAL:
            self._fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start_time = current_time
            
        return self._fps

    def detect_object(self, frame):
        try:
            # Start inference timing
            inference_start = time.perf_counter()
            
            if frame is None or len(frame.shape) != 3:
                return None
                
            # Preprocess image
            input_data = self._preprocess(frame)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get detection results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            
            # Record pure inference time
            self._last_inference_time = time.perf_counter() - inference_start
            
            # Find best detection
            max_idx = np.argmax(scores)
            max_score = scores[max_idx]
            
            # Check confidence threshold
            if max_score < self.confidence_threshold:
                return None
                
            # Calculate coordinates using numpy for efficiency
            h, w = frame.shape[:2]
            box = boxes[max_idx]
            coords = np.array([box[1] * w, box[0] * h, 
                             (box[3] - box[1]) * w, 
                             (box[2] - box[0]) * h])
            coords = np.clip(coords, 0, [w, h, w-coords[0], h-coords[1]])
            
            detected_class = self.labels[int(classes[max_idx])]
            
            return {
                'bounds': tuple(coords.astype(int)),
                'class': detected_class,
                'score': float(max_score),
                'inference_time': self._last_inference_time
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

    @property
    def inference_fps(self):
        """Get FPS based purely on inference time"""
        return 1.0 / self._last_inference_time if self._last_inference_time > 0 else 0