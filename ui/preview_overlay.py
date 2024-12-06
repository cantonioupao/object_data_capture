from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.animation import Animation
import numpy as np

class CapturePreviewOverlay(Widget):
    """
    Advanced overlay showing capture coverage and guidance.
    Visualizes captured angles and suggests next positions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture_zones = []
        self.current_pose = {'azimuth': 0, 'elevation': 0}
        self.highlight_animation = None
        
    def update(self, capture_zones, current_pose):
        """Updates the visualization with current capture status"""
        self.capture_zones = capture_zones
        self.current_pose = current_pose
        self.draw_coverage_map()
        
    def draw_coverage_map(self):
        """Draws a spherical coverage map showing captured angles"""
        self.canvas.clear()
        with self.canvas:
            # Draw spherical projection grid
            self._draw_sphere_grid()
            
            # Draw captured zones
            self._draw_captured_zones()
            
            # Draw current position indicator
            self._draw_position_indicator()
            
            # Draw next target position
            self._draw_next_target()
    
    def _draw_sphere_grid(self):
        """Draws a grid representing the capture sphere"""
        with self.canvas:
            Color(0.5, 0.5, 0.5, 0.3)
            
            # Draw latitude lines
            for elevation in range(-90, 91, 30):
                points = []
                for azimuth in range(0, 361, 10):
                    x, y = self._sphere_to_screen(azimuth, elevation)
                    points.extend([x, y])
                Line(points=points)
            
            # Draw longitude lines
            for azimuth in range(0, 361, 45):
                points = []
                for elevation in range(-90, 91, 5):
                    x, y = self._sphere_to_screen(azimuth, elevation)
                    points.extend([x, y])
                Line(points=points)
    
    def _draw_captured_zones(self):
        """Visualizes which zones have been captured"""
        with self.canvas:
            for zone in self.capture_zones:
                Color(0, 1, 0, 0.3 if zone.is_captured else 0.1)
                x, y = self._sphere_to_screen(zone.azimuth, zone.elevation)
                Ellipse(pos=(x-5, y-5), size=(10, 10))
    
    def _draw_position_indicator(self):
        """Shows current camera position"""
        x, y = self._sphere_to_screen(
            self.current_pose['azimuth'],
            self.current_pose['elevation']
        )
        with self.canvas:
            Color(1, 0, 0, 1)
            Line(circle=(x, y, 8))
            Line(circle=(x, y, 2))
    
    def _draw_next_target(self):
        """Highlights the next recommended capture position"""
        uncaptured = [z for z in self.capture_zones if not z.is_captured]
        if uncaptured:
            next_zone = min(uncaptured, key=lambda z: self._calculate_distance(
                z.azimuth, z.elevation,
                self.current_pose['azimuth'],
                self.current_pose['elevation']
            ))
            
            x, y = self._sphere_to_screen(next_zone.azimuth, next_zone.elevation)
            with self.canvas:
                Color(1, 1, 0, 1)
                Line(circle=(x, y, 12), width=2)
            
            # Animate the target indicator
            if not self.highlight_animation:
                target = Widget(pos=(x-15, y-15), size=(30, 30))
                self.add_widget(target)
                self.highlight_animation = Animation(
                    size=(40, 40),
                    pos=(x-20, y-20),
                    duration=1
                ) + Animation(
                    size=(30, 30),
                    pos=(x-15, y-15),
                    duration=1
                )
                self.highlight_animation.repeat = True
                self.highlight_animation.start(target)
    
    def _sphere_to_screen(self, azimuth: float, elevation: float) -> tuple:
        """Converts spherical coordinates to screen position"""
        # Use equirectangular projection
        x = self.width * (azimuth % 360) / 360
        y = self.height * (elevation + 90) / 180
        return x, y
    
    def _calculate_distance(self, az1, el1, az2, el2):
        """Calculates spherical distance between two points"""
        az1, az2 = np.radians([az1, az2])
        el1, el2 = np.radians([el1, el2])
        
        return np.arccos(
            np.sin(el1) * np.sin(el2) +
            np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
        )