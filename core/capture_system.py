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
    def __init__(self, object_detector, quality_analyzer, pose_estimator, storage_manager, config):
        self.object_detector = object_detector
        self.quality_analyzer = quality_analyzer
        self.pose_estimator = pose_estimator
        self.storage_manager = storage_manager
        self.capture_zones = self._initialize_capture_zones()
        self.captured_count = 0
        self.config = config

    def _initialize_capture_zones(self) -> List[CaptureZone]:
        """Creates positions around object that need to be photographed"""
        zones = []
        # Create zones every 45 degrees horizontally at three heights
        for azimuth in range(0, 360, 45):
            for elevation in [-30, 0, 30]:
                zones.append(CaptureZone(azimuth, elevation))
        return zones

    def _update_captured_zones(self, current_pose: Dict) -> None:
        """Updates which zones have been captured based on current camera position"""
        for zone in self.capture_zones:
            if not zone.is_captured:
                distance = self._calculate_pose_distance(
                    current_pose,
                    {"azimuth": zone.azimuth, "elevation": zone.elevation}
                )
                if distance < self.config.ANGLE_THRESHOLD:
                    zone.is_captured = True

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Processes a single frame through the complete pipeline:
        1. Object detection
        2. Position validation
        3. Quality analysis
        4. Pose estimation
        Returns status and guidance information.
        """
        # First, detect the object in the frame
        object_data = self.object_detector.detect_object(frame)
        if object_data is None:
            return {
                'status': 'no_object',
                'message': 'No object detected in frame',
                'progress': self.get_progress()
            }

        # Check if object is properly positioned
        if not self._validate_object_position(object_data):
            return {
                'status': 'bad_position',
                'message': 'Please center the object in frame',
                'progress': self.get_progress()
            }

        # Analyze image quality
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

        # Check if we've captured all zones
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
            'next_zone': next_zone,
            'object_data': object_data
        }

    def _validate_object_position(self, object_data: Dict) -> bool:
        """
        Checks if the detected object is properly positioned and sized in the frame.
        Returns True if the object meets all criteria.
        """
        bounds = object_data['bounds']
        center = object_data['center']

        # Calculate object size relative to frame
        width = bounds['x_max'] - bounds['x_min']
        height = bounds['y_max'] - bounds['y_min']
        frame_width = self.config.CAMERA_RESOLUTION[0]
        frame_height = self.config.CAMERA_RESOLUTION[1]

        size_ratio = (width * height) / (frame_width * frame_height)
        if size_ratio < self.config.MIN_OBJECT_SIZE:
            return False

        # Check if object is centered in frame
        center_x = center['x'] / frame_width
        center_y = center['y'] / frame_height
        return (0.4 < center_x < 0.6) and (0.4 < center_y < 0.6)

    def capture(self, frame: np.ndarray) -> bool:
        """
        Attempts to capture and save a frame if it meets all quality criteria.
        Returns True if capture was successful.
        """
        try:
            # Check object detection and positioning
            object_data = self.object_detector.detect_object(frame)
            if object_data is None or not self._validate_object_position(object_data):
                return False

            # Get pose and quality information
            pose = self.pose_estimator.get_pose(frame)
            quality = self.quality_analyzer.analyze_frame(frame)

            # Prepare metadata for storage
            metadata = {
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "pose": pose,
                "quality_metrics": quality.__dict__,
                "object_data": object_data
            }

            # Save capture and update status
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
        """
        Calculates the angular distance between two camera poses.
        Takes into account wraparound at 360 degrees.
        """
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
        """Calculates overall capture completion percentage"""
        captured = sum(1 for zone in self.capture_zones if zone.is_captured)
        return (captured / len(self.capture_zones)) * 100