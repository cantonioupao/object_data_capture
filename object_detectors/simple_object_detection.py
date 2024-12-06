import numpy as np
from core.base import ObjectDetector

class SimpleObjectDetector(ObjectDetector):
    def __init__(self):
        self.background = None
        self.prev_frame = None
        
    def detect_object(self, frame):
        if self.background is None:
            self.background = frame
            return None
            
        # Motion detection
        motion_mask = self._detect_motion(frame)
        # Background subtraction
        bg_mask = self._subtract_background(frame)
        # Combine masks
        mask = motion_mask & bg_mask
        
        if not np.any(mask):
            return None
            
        # Object properties
        return {
            'bounds': self._get_bounds(mask),
            'center': self._get_center(mask),
            'contour': self._get_contour(mask),
            'features': self._extract_features(frame, mask)
        }
        
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