import numpy as np
from typing import Dict, Tuple

def calculate_spherical_distance(angle1: Dict[str, float], angle2: Dict[str, float]) -> float:
    """
    Calculates the spherical distance between two angles on a unit sphere.
    This gives us more accurate distance measurements for camera positions.
    
    Args:
        angle1: Dictionary with 'azimuth' and 'elevation' in degrees
        angle2: Dictionary with 'azimuth' and 'elevation' in degrees
        
    Returns:
        float: Distance between the angles in degrees
    """
    # Convert to radians for calculations
    az1 = np.radians(angle1['azimuth'])
    el1 = np.radians(angle1['elevation'])
    az2 = np.radians(angle2['azimuth'])
    el2 = np.radians(angle2['elevation'])
    
    # Convert spherical to cartesian coordinates
    x1 = np.cos(el1) * np.cos(az1)
    y1 = np.cos(el1) * np.sin(az1)
    z1 = np.sin(el1)
    
    x2 = np.cos(el2) * np.cos(az2)
    y2 = np.cos(el2) * np.sin(az2)
    z2 = np.sin(el2)
    
    # Calculate great circle distance
    distance = np.arccos(np.clip(x1*x2 + y1*y2 + z1*z2, -1.0, 1.0))
    return np.degrees(distance)

def normalize_angle(angle: float) -> float:
    """
    Normalizes an angle to be between 0 and 360 degrees.
    Useful for consistent angle comparisons.
    """
    return angle % 360

def get_movement_direction(current: float, target: float, threshold: float = 15.0) -> str:
    """
    Determines the shortest direction to move from current to target angle.
    
    Args:
        current: Current angle in degrees
        target: Target angle in degrees
        threshold: Minimum difference to suggest movement
        
    Returns:
        str: Movement direction ('left', 'right', or 'hold')
    """
    diff = normalize_angle(target - current)
    if abs(diff) <= threshold:
        return 'hold'
    return 'right' if diff < 180 else 'left'