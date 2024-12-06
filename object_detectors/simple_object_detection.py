import numpy as np
from core.base import ObjectDetector

class SimpleObjectDetector(ObjectDetector):
    def __init__(self):
        self.background = None
        self.prev_frame = None

    def detect_object(self, frame):
        # Convert to grayscale
        gray = np.mean(frame, axis=2)
        
        # Detect non-white objects (white is close to 255)
        mask = gray < 150  # Threshold for non-white pixels
        
        if not np.any(mask):
            return None
            
        y_coords, x_coords = np.where(mask)
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords) 
        y_max = np.max(y_coords)

        width = x_max - x_min
        height = y_max - y_min

        return {
        'bounds': (x_min, y_min, width, height),
        'center': {'x': (x_min + x_max) // 2, 'y': (y_min + y_max) // 2},
        'mask': mask
        }
        
       
    def _get_contour(self, mask):
        # Edge detection
        contour = np.zeros_like(mask)
        contour[1:] = mask[1:] != mask[:-1]  # Vertical edges
        contour[:,1:] |= mask[:,1:] != mask[:,:-1]  # Horizontal edges
        return np.where(contour)
        
    def _detect_motion(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            return np.ones_like(frame[:,:,0], dtype=bool)
        
        diff = np.abs(frame - self.prev_frame)
        self.prev_frame = frame
        return np.mean(diff, axis=2) > 30
        
    def _subtract_background(self, frame):
        diff = np.abs(frame - self.background)
        return np.mean(diff, axis=2) > 30
        
    def _get_bounds(self, mask):
        y_idx, x_idx = np.where(mask)
        return {
            'x_min': np.min(x_idx),
            'x_max': np.max(x_idx),
            'y_min': np.min(y_idx),
            'y_max': np.max(y_idx)
        }
        
    def _get_center(self, mask):
        y_idx, x_idx = np.where(mask)
        return {
            'x': np.mean(x_idx),
            'y': np.mean(y_idx)
        }
        
    def _get_contour(self, mask):
        # Simple edge detection on mask
        edges = np.zeros_like(mask)
        edges[1:] = mask[1:] != mask[:-1]
        edges[:,1:] |= mask[:,1:] != mask[:,:-1]
        return np.where(edges)
        
    def _extract_features(self, frame, mask):
        gray = np.mean(frame, axis=2)
        features = []
        
        for y in range(3, gray.shape[0]-3):
            for x in range(3, gray.shape[1]-3):
                if mask[y,x]:
                    window = gray[y-3:y+4, x-3:x+4]
                    if self._is_feature(window):
                        features.append([x, y])
                        
        return np.array(features)
        
    def _is_feature(self, window):
        dx = np.sum(np.abs(np.diff(window, axis=1)))
        dy = np.sum(np.abs(np.diff(window, axis=0)))
        return dx > 50 and dy > 50  # Threshold for corner detection