from abc import ABC, abstractmethod
import numpy as np
from .data_types import CaptureMetrics
from typing import Dict, List

class QualityAnalyzer(ABC):
    @abstractmethod
    def analyze_frame(self, frame: np.ndarray) -> CaptureMetrics:
        pass

class ObjectDetector(ABC):
    @abstractmethod
    def detect_object(self, frame: np.ndarray) -> Dict:
        pass

class PoseEstimator(ABC):
    @abstractmethod
    def get_pose(self, frame: np.ndarray) -> Dict[str, float]:
        pass

class ThreeDReconstructor(ABC):
    @abstractmethod
    def reconstruct_object(self, frames: List[np.ndarray]) -> np.ndarray:
        pass

class StorageManager(ABC):
    @abstractmethod
    def save_capture(self, frame: np.ndarray, metadata: Dict) -> bool:
        pass
    
    @abstractmethod
    def get_captures(self) -> List[Dict]:
        pass