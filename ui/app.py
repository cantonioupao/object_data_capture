from kivy.app import App
from kivy.uix.relativelayout import RelativeLayout 
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock
from kivy.core.window import Window
import cv2
import numpy as np
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty

class CaptureGuideOverlay(Widget):
    """
    Provides visual guidance overlays on the camera preview.
    Shows target positions, current orientation, and capture zones.
    """
    capture_progress = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture_zones = []
        self.current_pose = {'azimuth': 0, 'elevation': 0}
        
    def update_guidance(self, capture_zones, current_pose):
        """Updates the visual guidance based on current status"""
        self.capture_zones = capture_zones
        self.current_pose = current_pose
        self.draw_guidance()
    
    def draw_guidance(self):
        """Draws all guidance elements on screen"""
        self.canvas.clear()
        with self.canvas:
            # Draw alignment grid
            Color(0, 1, 0, 0.5)  # Semi-transparent green
            
            # Center crosshair
            center_x = self.width / 2
            center_y = self.height / 2
            Line(points=[center_x - 20, center_y, center_x + 20, center_y])
            Line(points=[center_x, center_y - 20, center_x, center_y + 20])
            
            # Draw orientation compass
            self.draw_compass(center_x - 50, 50, 40)
            
            # Draw capture progress indicator
            self.draw_progress_wheel(center_x + 50, 50, 40)
    
    def draw_compass(self, x, y, radius):
        """Draws a compass showing current orientation"""
        with self.canvas:
            Color(1, 1, 1, 0.8)
            Line(circle=(x, y, radius))
            
            # Draw current orientation indicator
            angle = np.radians(self.current_pose['azimuth'])
            end_x = x + radius * np.cos(angle)
            end_y = y + radius * np.sin(angle)
            Color(0, 1, 0, 1)
            Line(points=[x, y, end_x, end_y], width=2)
    
    def draw_progress_wheel(self, x, y, radius):
        """Shows capture progress as a circular indicator"""
        completed = sum(1 for zone in self.capture_zones if zone.is_captured)
        progress = completed / len(self.capture_zones)
        
        with self.canvas:
            # Draw progress arc
            Color(0, 1, 0, 0.8)
            Line(circle=(x, y, radius), width=2)
            
            # Fill completed portion
            if progress > 0:
                angles = np.linspace(0, progress * 360, 30)
                points = []
                for angle in angles:
                    rad = np.radians(angle)
                    points.extend([
                        x + radius * np.cos(rad),
                        y + radius * np.sin(rad)
                    ])
                Line(points=points, width=2)

class CaptureInterface(BoxLayout):
    """
    Main interface for the capture application.
    Combines camera preview, guidance overlay, and controls.
    """
    def __init__(self, capture_system, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.capture_system = capture_system
        
        # Create main preview container
        preview_container = RelativeLayout(size_hint_y=0.8)
        
        # Camera preview
        self.preview = Widget()
        preview_container.add_widget(self.preview)
        
        # Guidance overlay
        self.overlay = CaptureGuideOverlay()
        preview_container.add_widget(self.overlay)
        
        self.add_widget(preview_container)
        
        # Status and controls
        controls = BoxLayout(orientation='vertical', size_hint_y=0.2)
        self.status_label = Label(text='Initializing...', size_hint_y=0.5)
        self.capture_button = Button(
            text='Capture',
            size_hint_y=0.5,
            background_color=(0.3, 0.8, 0.3, 1)
        )
        self.capture_button.bind(on_press=self.capture_image)
        
        controls.add_widget(self.status_label)
        controls.add_widget(self.capture_button)
        self.add_widget(controls)
        
        self.setup_camera()
    
    def setup_camera(self):
        """Initializes camera capture and preview updates"""
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_preview, 1.0/30.0)
    
    def update_preview(self, dt):
        """Updates camera preview and guidance overlay"""
        ret, frame = self.capture.read()
        if ret:
            # Process frame
            result = self.capture_system.process_frame(frame)
            self.status_label.text = result['message']
            
            # Update capture button state
            self.capture_button.disabled = result['status'] != 'ready'
            
            # Update guidance overlay
            self.overlay.update_guidance(
                self.capture_system.capture_zones,
                result.get('current_pose', {'azimuth': 0, 'elevation': 0})
            )
            
            # Convert frame for display
            buf = cv2.flip(frame, 0)
            buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
            texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]),
                colorfmt='rgb'
            )
            texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            
            # Update preview
            self.preview.canvas.clear()
            with self.preview.canvas:
                Rectangle(texture=texture, size=self.preview.size)
    
    def capture_image(self, instance):
        """Handles image capture button press"""
        ret, frame = self.capture.read()
        if ret:
            success = self.capture_system.capture(frame)
            if success:
                self.status_label.text = "Capture successful!"
            else:
                self.status_label.text = "Capture failed - please try again"

class DamageCaptureApp(App):
    """
    Main application class.
    Sets up and manages the capture interface.
    """
    def __init__(self, capture_system, **kwargs):
        super().__init__(**kwargs)
        self.capture_system = capture_system
    
    def build(self):
        """Creates and returns the main interface"""
        return CaptureInterface(self.capture_system)