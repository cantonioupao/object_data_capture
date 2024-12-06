from typing import Dict, List

class CaptureConfig:
    """
    Central configuration for the capture system.
    Makes it easy to adjust parameters without changing code throughout the app.
    """
    
    # Camera capture settings
    CAMERA_RESOLUTION = (1280, 720)
    CAMERA_FPS = 30
    
    # Quality thresholds
    MIN_BRIGHTNESS = 0.3
    MAX_BRIGHTNESS = 0.8
    MIN_CONTRAST = 0.5
    MAX_BLUR = 100
    MIN_OBJECT_SIZE = 0.3  # As fraction of frame
    
    # Capture zone settings
    AZIMUTH_STEPS = 45     # Degrees between horizontal captures
    ELEVATION_LEVELS = [-30, 0, 30]  # Vertical angles to capture
    ANGLE_THRESHOLD = 15.0  # Maximum deviation from target angle
    
    # Storage settings
    IMAGE_FORMAT = 'jpg'
    IMAGE_QUALITY = 95
    DB_PATH = 'captures.db'
    
    @classmethod
    def get_quality_thresholds(cls) -> Dict[str, float]:
        """Returns all quality-related thresholds as a dictionary"""
        return {
            'min_brightness': cls.MIN_BRIGHTNESS,
            'max_brightness': cls.MAX_BRIGHTNESS,
            'min_contrast': cls.MIN_CONTRAST,
            'max_blur': cls.MAX_BLUR,
            'min_object_size': cls.MIN_OBJECT_SIZE
        }
    
    @classmethod
    def get_capture_angles(cls) -> List[Dict[str, float]]:
        """Generates all required capture angles based on configuration"""
        angles = []
        for elevation in cls.ELEVATION_LEVELS:
            for azimuth in range(0, 360, cls.AZIMUTH_STEPS):
                angles.append({
                    'azimuth': float(azimuth),
                    'elevation': float(elevation)
                })
        return angles