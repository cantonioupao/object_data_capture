import numpy as np
from PyQt6.QtWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class CaptureZoneVisualizer(QOpenGLWidget):
    """
    Creates an interactive 3D visualization of the capture zones around an object.
    Shows captured and uncaptured regions in real-time as the user moves the camera.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)  # Reasonable size for visualization
        
        # Camera/view control variables
        self.camera_distance = 5.0
        self.rotation = [30.0, 0.0, 0.0]  # Initial view angles
        self.last_pos = None
        
        # Visualization state
        self.capture_zones = []
        self.current_pose = None
        self.highlighted_zone = None
        
    def initializeGL(self):
        """Sets up the OpenGL rendering context."""
        glutInit()  # Initialize GLUT for basic shapes
        
        # Enable 3D rendering features
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Set background color to light gray
        glClearColor(0.9, 0.9, 0.9, 1.0)

    def paintGL(self):
        """Renders the 3D scene with capture zones and object."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Position camera
        gluLookAt(0, 0, self.camera_distance,  # Camera position
                 0, 0, 0,                      # Look at center
                 0, 1, 0)                      # Up direction
        
        # Apply view rotation
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        
        # Draw capture sphere and zones
        self._draw_capture_zones()
        
        # Draw object placeholder
        self._draw_object()
        
        # Draw current camera position if available
        if self.current_pose:
            self._draw_camera_position()

    def _draw_capture_zones(self):
        """Draws all capture zones as colored spheres on a transparent guide sphere."""
        # Draw guide sphere
        glPushMatrix()
        glColor4f(0.8, 0.8, 0.8, 0.2)  # Transparent gray
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glutWireSphere(2.0, 20, 20)     # Radius 2.0, 20 segments
        glDisable(GL_BLEND)
        glPopMatrix()
        
        # Draw each capture zone
        for zone in self.capture_zones:
            self._draw_zone(zone)

    def _draw_zone(self, zone):
        """Draws a single capture zone with appropriate color and highlighting."""
        glPushMatrix()
        
        # Convert zone's spherical coordinates to Cartesian
        theta = np.radians(zone.azimuth)
        phi = np.radians(zone.elevation)
        x = 2.0 * np.cos(phi) * np.cos(theta)  # Radius 2.0
        y = 2.0 * np.sin(phi)
        z = 2.0 * np.cos(phi) * np.sin(theta)
        
        glTranslatef(x, y, z)
        
        # Set color based on capture status and highlighting
        if zone is self.highlighted_zone:
            glColor3f(1.0, 1.0, 0.0)  # Yellow for highlighted
        elif zone.is_captured:
            glColor3f(0.0, 0.8, 0.0)  # Green for captured
        else:
            glColor3f(0.8, 0.0, 0.0)  # Red for uncaptured
            
        glutSolidSphere(0.1, 10, 10)  # Small sphere for zone
        glPopMatrix()

    def _draw_object(self):
        """Draws a simple placeholder for the captured object."""
        glPushMatrix()
        glColor3f(0.5, 0.5, 0.5)  # Medium gray
        glutSolidCube(0.5)        # Small cube as placeholder
        glPopMatrix()

    def _draw_camera_position(self):
        """Draws an indicator for current camera position."""
        glPushMatrix()
        
        # Convert current pose to position
        theta = np.radians(self.current_pose['azimuth'])
        phi = np.radians(self.current_pose['elevation'])
        x = 2.0 * np.cos(phi) * np.cos(theta)
        y = 2.0 * np.sin(phi)
        z = 2.0 * np.cos(phi) * np.sin(theta)
        
        glTranslatef(x, y, z)
        glColor3f(0.0, 0.0, 1.0)  # Blue for camera
        
        # Draw camera frustum
        self._draw_camera_frustum()
        glPopMatrix()

    def _draw_camera_frustum(self):
        """Draws a simplified camera frustum to show orientation."""
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(-0.2, -0.2, -0.5)
        glVertex3f(0, 0, 0)
        glVertex3f(0.2, -0.2, -0.5)
        glVertex3f(0, 0, 0)
        glVertex3f(0.2, 0.2, -0.5)
        glVertex3f(0, 0, 0)
        glVertex3f(-0.2, 0.2, -0.5)
        glEnd()

    def update_state(self, zones, current_pose=None):
        """Updates visualization with new capture zones and camera pose."""
        self.capture_zones = zones
        self.current_pose = current_pose
        self.update()  # Trigger redraw

    def mousePressEvent(self, event):
        """Handles mouse press for rotation control."""
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handles mouse drag for view rotation."""
        if self.last_pos is None:
            return
            
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        # Update rotation based on mouse movement
        self.rotation[0] += dy * 0.5
        self.rotation[1] += dx * 0.5
        
        self.last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        """Handles mouse wheel for zoom control."""
        delta = event.angleDelta().y()
        self.camera_distance -= delta * 0.001
        self.camera_distance = max(3.0, min(10.0, self.camera_distance))
        self.update()