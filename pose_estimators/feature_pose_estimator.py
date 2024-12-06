import numpy as np
from core.base import PoseEstimator

class FeaturePoseEstimator(PoseEstimator):
    def __init__(self):
        self.prev_features = None
        self.current_pose = {'azimuth': 0.0, 'elevation': 0.0}

    def get_pose(self, frame: np.ndarray) -> dict:
        features = self._extract_features(frame)
        
        if self.prev_features is not None:
            rotation = self._estimate_rotation(features)
            self.current_pose['azimuth'] = (self.current_pose['azimuth'] + rotation[1]) % 360
            self.current_pose['elevation'] = max(min(
                self.current_pose['elevation'] + rotation[0], 90), -90)
        
        self.prev_features = features
        return self.current_pose.copy()

    def _extract_features(self, frame):
        gray = np.mean(frame, axis=2)
        features = []
        for y in range(3, gray.shape[0]-3):
            for x in range(3, gray.shape[1]-3):
                window = gray[y-3:y+4, x-3:x+4]
                if self._is_feature(window):
                    features.append([x, y])
        return np.array(features)

    def _is_feature(self, window):
        dx = np.sum(np.abs(np.diff(window, axis=1)))
        dy = np.sum(np.abs(np.diff(window, axis=0)))
        return dx > 50 and dy > 50

    def _estimate_rotation(self, curr_features):
        if len(curr_features) < 4 or len(self.prev_features) < 4:
            return [0, 0]

        prev_centroid = np.mean(self.prev_features, axis=0)
        curr_centroid = np.mean(curr_features, axis=0)
        
        translation = curr_centroid - prev_centroid
        return [translation[1] * 0.1, translation[0] * 0.1]  