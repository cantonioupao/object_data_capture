# visualization/basic_visualizer.py
import numpy as np
import cv2

class BasicVisualizer:
    """
    A simple 3D visualization system built from scratch.
    Uses basic geometry and projection to create a visual representation
    of capture zones around an object.
    """
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        # Define a virtual sphere around the object
        self.sphere_radius = min(frame_width, frame_height) // 4
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2

    def project_3d_to_2d(self, azimuth, elevation):
        """
        Converts 3D spherical coordinates to 2D screen coordinates.
        This simulates how points in 3D space would appear on a 2D screen.
        """
        # Convert angles to radians for mathematical operations
        theta = np.radians(azimuth)
        phi = np.radians(elevation)
        
        # Convert spherical to Cartesian coordinates
        x = self.sphere_radius * np.cos(phi) * np.cos(theta)
        y = self.sphere_radius * np.sin(phi)
        z = self.sphere_radius * np.cos(phi) * np.sin(theta)
        
        # Apply perspective projection
        scale = 1.0 + (z / (2.0 * self.sphere_radius))
        screen_x = self.center_x + int(x * scale)
        screen_y = self.center_y - int(y * scale)
        
        return screen_x, screen_y, scale

    def draw_capture_zones(self, frame, zones):
        """
        Draws the capture zones onto the camera frame.
        Creates a visual overlay showing captured and uncaptured areas.
        """
        # Create a copy of the frame to draw on
        visualization = frame.copy()
        
        # Draw reference sphere grid
        self._draw_reference_grid(visualization)
        
        # Draw each capture zone
        for zone in zones:
            x, y, scale = self.project_3d_to_2d(zone.azimuth, zone.elevation)
            
            # Size of indicator varies with depth
            radius = int(10 * scale)
            
            # Color depends on capture status
            color = (0, 255, 0) if zone.is_captured else (0, 0, 255)
            
            # Draw the zone indicator
            cv2.circle(visualization, (x, y), radius, color, -1)
            
            # Add depth effect
            alpha = 0.7 * scale  # Transparency based on depth
            cv2.circle(visualization, (x, y), radius, (255, 255, 255), 1)

        return visualization

    def _draw_reference_grid(self, frame):
        """
        Draws a reference grid to help visualize the 3D space.
        Creates a sphere-like wireframe effect.
        """
        # Draw horizontal circles at different elevations
        for elevation in [-60, -30, 0, 30, 60]:
            points = []
            for azimuth in range(0, 361, 10):
                x, y, _ = self.project_3d_to_2d(azimuth, elevation)
                points.append((x, y))
            
            # Connect points to form circles
            for i in range(len(points) - 1):
                pt1 = points[i]
                pt2 = points[i + 1]
                cv2.line(frame, pt1, pt2, (128, 128, 128), 1)

        # Draw vertical lines
        for azimuth in range(0, 360, 45):
            points = []
            for elevation in range(-90, 91, 10):
                x, y, _ = self.project_3d_to_2d(azimuth, elevation)
                points.append((x, y))
            
            for i in range(len(points) - 1):
                pt1 = points[i]
                pt2 = points[i + 1]
                cv2.line(frame, pt1, pt2, (128, 128, 128), 1)