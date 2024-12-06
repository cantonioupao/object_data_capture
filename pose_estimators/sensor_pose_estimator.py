from typing import Dict, Optional
import numpy as np
from kivy.clock import Clock
from plyer import accelerometer, gyroscope
from utils.angle_utils import normalize_angle

class SensorPoseEstimator:
    """
    Estimates camera pose using device sensors for accurate positioning.
    Combines accelerometer and gyroscope data for stable measurements.
    """
    def __init__(self):
        self.last_accel = {'x': 0, 'y': 0, 'z': 0}
        self.last_gyro = {'x': 0, 'y': 0, 'z': 0}
        self.current_azimuth = 0.0
        self.current_elevation = 0.0
        
        # Start sensor updates
        try:
            accelerometer.enable()
            gyroscope.enable()
            Clock.schedule_interval(self._update_sensors, 1.0/30.0)  # 30 Hz updates
        except:
            print("Warning: Sensor access failed, falling back to basic estimation")
    
    def _update_sensors(self, dt):
        """
        Updates pose estimation using latest sensor data.
        Uses complementary filter to combine accelerometer and gyroscope data.
        """
        try:
            # Get latest sensor readings
            accel_data = accelerometer.acceleration
            gyro_data = gyroscope.rotation
            
            # Update stored values
            self.last_accel = {
                'x': accel_data[0],
                'y': accel_data[1],
                'z': accel_data[2]
            }
            
            self.last_gyro = {
                'x': gyro_data[0],
                'y': gyro_data[1],
                'z': gyro_data[2]
            }
            
            # Calculate angles from accelerometer
            accel_elevation = np.degrees(np.arctan2(
                self.last_accel['y'],
                np.sqrt(self.last_accel['x']**2 + self.last_accel['z']**2)
            ))
            
            accel_azimuth = np.degrees(np.arctan2(
                self.last_accel['x'],
                self.last_accel['z']
            ))
            
            # Combine with gyroscope data using complementary filter
            alpha = 0.96  # Filter coefficient
            self.current_elevation = alpha * (self.current_elevation + self.last_gyro['y'] * dt) + \
                                   (1 - alpha) * accel_elevation
            
            self.current_azimuth = normalize_angle(
                alpha * (self.current_azimuth + self.last_gyro['z'] * dt) + \
                (1 - alpha) * accel_azimuth
            )
            
        except:
            pass
    
    def get_pose(self, frame: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Returns current camera pose based on sensor data.
        Frame parameter is optional and only used as fallback.
        """
        return {
            'azimuth': self.current_azimuth,
            'elevation': self.current_elevation
        }
    
    def reset_orientation(self):
        """Resets the orientation reference point"""
        self.current_azimuth = 0.0
        self.current_elevation = 0.0