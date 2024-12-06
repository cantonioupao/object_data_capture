from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import time

@dataclass
class CaptureZone:
    """
    Represents a specific camera position for capturing the object.
    Think of this like marking points on an invisible sphere around the object.
    """
    azimuth: float    # Horizontal angle (0-360 degrees)
    elevation: float  # Vertical angle (-90 to 90 degrees)
    is_captured: bool = False

class DamageCaptureSystem:
    """
    Main system that coordinates capture process, quality checks, and storage.
    Works like an orchestra conductor, making sure all parts work together.
    """
    def __init__(self, quality_analyzer, pose_estimator, storage_manager):
        self.quality_analyzer = quality_analyzer
        self.pose_estimator = pose_estimator
        self.storage_manager = storage_manager
        self.capture_zones = self._initialize_capture_zones()
        self.captured_count = 0
        
    def _initialize_capture_zones(self) -> List[CaptureZone]:
        """Creates positions around object that need to be photographed"""
        zones = []
        # Create zones every 45 degrees horizontally at three heights
        for azimuth in range(0, 360, 45):
            for elevation in [-30, 0, 30]:
                zones.append(CaptureZone(azimuth, elevation))
        return zones
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyzes current camera view and provides guidance.
        Returns status information and user instructions.
        """
        # Check image quality
        quality_result = self.quality_analyzer.analyze_frame(frame)
        if not quality_result.is_acceptable:
            return {
                'status': 'quality_issue',
                'message': quality_result.message,
                'progress': self.get_progress()
            }
            
        # Get current camera position
        pose = self.pose_estimator.get_pose(frame)
        next_zone = self._find_nearest_uncaptured_zone(pose)
        
        if not next_zone:
            return {
                'status': 'complete',
                'message': 'All angles captured!',
                'progress': 100.0
            }
            
        # Generate guidance for reaching next position
        guidance = self._generate_movement_guidance(pose, next_zone)
        return {
            'status': 'in_progress',
            'message': guidance,
            'progress': self.get_progress(),
            'next_zone': next_zone
        }
    
    def capture(self, frame: np.ndarray) -> bool:
        """
        Saves a frame if it completes a new capture zone.
        Returns True if capture was successful.
        """
        try:
            pose = self.pose_estimator.get_pose(frame)
            quality = self.quality_analyzer.analyze_frame(frame)
            
            metadata = {
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "pose": pose,
                "quality_metrics": quality.__dict__
            }
            
            # Try to save the capture
            if self.storage_manager.save_capture(frame, metadata):
                self._update_captured_zones(pose)
                return True
                
            return False
            
        except Exception as e:
            print(f"Capture error: {e}")
            return False
    
    def _find_nearest_uncaptured_zone(self, current_pose: Dict) -> Optional[CaptureZone]:
        """Finds the closest zone that still needs to be captured"""
        min_distance = float('inf')
        nearest_zone = None
        
        for zone in self.capture_zones:
            if not zone.is_captured:
                distance = self._calculate_pose_distance(
                    current_pose, 
                    {"azimuth": zone.azimuth, "elevation": zone.elevation}
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest_zone = zone
                    
        return nearest_zone
    
    def _calculate_pose_distance(self, pose1: Dict, pose2: Dict) -> float:
        """Calculates how far apart two camera positions are"""
        az_diff = min(
            abs(pose1["azimuth"] - pose2["azimuth"]),
            360 - abs(pose1["azimuth"] - pose2["azimuth"])
        )
        el_diff = abs(pose1["elevation"] - pose2["elevation"])
        return np.sqrt(az_diff**2 + el_diff**2)
    
    def _generate_movement_guidance(self, current: Dict, target: CaptureZone) -> str:
        """Creates user-friendly instructions for camera movement"""
        az_diff = target.azimuth - current["azimuth"]
        el_diff = target.elevation - current["elevation"]
        
        directions = []
        if abs(az_diff) > 15:
            directions.append("right" if az_diff > 0 else "left")
        if abs(el_diff) > 15:
            directions.append("up" if el_diff > 0 else "down")
            
        return f"Move camera {' and '.join(directions)}" if directions else "Hold position"
    
    def get_progress(self) -> float:
        """Calculates completion percentage"""
        captured = sum(1 for zone in self.capture_zones if zone.is_captured)
        return (captured / len(self.capture_zones)) * 100