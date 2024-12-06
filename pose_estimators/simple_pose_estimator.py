import numpy as np
from core.base import PoseEstimator
from typing import Dict

class BasicPoseEstimator(PoseEstimator):
    def __init__(self):
        self.last_angle = 0
        
    def get_pose(self, frame: np.ndarray) -> Dict[str, float]:
        # Mock implementation - replace with actual pose estimation
        self.last_angle = (self.last_angle + 1) % 360
        return {
            "azimuth": self.last_angle,
            "elevation": 0
        }