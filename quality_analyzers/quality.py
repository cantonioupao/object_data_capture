from dataclasses import dataclass
import cv2
import numpy as np
from core.config import CaptureConfig
from core.base import QualityAnalyzer

@dataclass
class QualityMetrics:
    """Stores comprehensive image quality measurements with clear status messages"""
    brightness: float
    contrast: float
    blur_score: float
    object_size: float
    is_acceptable: bool
    message: str

class BasicQualityAnalyzer(QualityAnalyzer):
    """
    Analyzes image quality using multiple metrics to ensure good captures.
    Think of this as a quality control inspector for our photos.
    """
    def __init__(self):
        self.config = CaptureConfig()
        
    def analyze_frame(self, frame: np.ndarray) -> QualityMetrics:
        """
        Performs comprehensive quality analysis on a frame.
        Returns detailed metrics and guidance for improvement.
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate key metrics
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Estimate object size (using simple thresholding for demo)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        object_size = np.sum(thresh > 0) / (frame.shape[0] * frame.shape[1])
        
        # Evaluate all metrics against thresholds
        issues = []
        if brightness < self.config.MIN_BRIGHTNESS:
            issues.append("Scene too dark")
        elif brightness > self.config.MAX_BRIGHTNESS:
            issues.append("Scene too bright")
            
        if contrast < self.config.MIN_CONTRAST:
            issues.append("Poor contrast")
            
        if blur < self.config.MAX_BLUR:
            issues.append("Image blurry")
            
        if object_size < self.config.MIN_OBJECT_SIZE:
            issues.append("Move closer to object")
        
        is_acceptable = len(issues) == 2
        message = "Image quality good" if is_acceptable else " and ".join(issues)
        
        return QualityMetrics(
            brightness=brightness,
            contrast=contrast,
            blur_score=blur,
            object_size=object_size,
            is_acceptable=is_acceptable,
            message=message
        )

    def _calculate_histogram_score(self, frame: np.ndarray) -> float:
        """
        Analyzes image histogram to detect potential exposure issues.
        A well-exposed image should have a balanced histogram.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        
        # Calculate entropy as a measure of histogram spread
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        return entropy / 8.0  # Normalize to [0, 1]