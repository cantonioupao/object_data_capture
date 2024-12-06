import numpy as np
import tensorflow as tf
from core.base import ObjectDetector

class TFLiteDetector(ObjectDetector):
   def __init__(self, model_path='models/mobilenet_v1.tflite'):
       self.interpreter = tf.lite.Interpreter(model_path)
       self.interpreter.allocate_tensors()
       
       # Get input/output details
       self.input_details = self.interpreter.get_input_details()
       self.output_details = self.interpreter.get_output_details()
       
       self.input_shape = self.input_details[0]['shape']
       self.labels = self._load_labels()

   def detect_object(self, frame):
       # Preprocess image
       input_data = self._preprocess(frame)
       self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
       
       # Run inference
       self.interpreter.invoke()
       
       # Get results
       boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
       classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
       scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
       
       # Get highest confidence detection
       max_idx = np.argmax(scores)
       if scores[max_idx] < 0.5:
           return None
           
       # Convert normalized coords to pixel coords
       box = boxes[max_idx]
       h, w = frame.shape[:2]
       x = int(box[1] * w)
       y = int(box[0] * h)
       width = int((box[3] - box[1]) * w)
       height = int((box[2] - box[0]) * h)
       
       return {
           'bounds': (x, y, width, height),
           'class': self.labels[int(classes[max_idx])],
           'score': float(scores[max_idx])
       }

   def _preprocess(self, frame):
       # Resize
       frame = tf.image.resize(frame, (self.input_shape[1], self.input_shape[2]))
       # Normalize to [-1,1]
       frame = (frame - 127.5) / 127.5
       # Add batch dimension
       return np.expand_dims(frame, axis=0).astype(np.float32)

   def _load_labels(self):
       with open('models/labels.txt', 'r') as f:
           return [line.strip() for line in f.readlines()]