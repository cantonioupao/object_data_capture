from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class CaptureMetrics:
    quality_score: float
    blur_score: float
    brightness: float
    coverage_score: float

@dataclass
class CaptureResult:
    status: str
    message: str
    metrics: Optional[CaptureMetrics] = None
    pose: Optional[Dict[str, float]] = None

# core/base.py
from abc import ABC, abstractmethod
import numpy as np
from .data_types import CaptureMetrics

class QualityAnalyzer(ABC):
    @abstractmethod
    def analyze_frame(self, frame: np.ndarray) -> CaptureMetrics:
        pass

class PoseEstimator(ABC):
    @abstractmethod
    def get_pose(self, frame: np.ndarray) -> Dict[str, float]:
        pass

class StorageManager(ABC):
    @abstractmethod
    def save_capture(self, frame: np.ndarray, metadata: Dict) -> bool:
        pass
    
    @abstractmethod
    def get_captures(self) -> List[Dict]:
        pass