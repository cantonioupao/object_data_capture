from kivy.uix.modalview import ModalView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.animation import Animation
from kivy.clock import Clock

class CalibrationView(ModalView):
    """
    A calibration wizard that helps users set up accurate angle measurements.
    Think of this like calibrating a compass - it helps establish reference points.
    """
    def __init__(self, pose_estimator, **kwargs):
        super().__init__(**kwargs)
        self.pose_estimator = pose_estimator
        self.size_hint = (0.9, 0.9)
        self.setup_ui()
        
    def setup_ui(self):
        layout = BoxLayout(orientation='vertical', spacing='10dp', padding='20dp')
        
        # Instructions for calibration
        self.instruction_label = Label(
            text='Place your device on a flat surface\nand point it at the object',
            size_hint_y=0.6,
            halign='center'
        )
        layout.add_widget(self.instruction_label)
        
        # Calibration steps indicator
        self.progress_label = Label(
            text='Step 1/3',
            size_hint_y=0.2
        )
        layout.add_widget(self.progress_label)
        
        # Action button
        self.action_button = Button(
            text='Start Calibration',
            size_hint_y=0.2,
            on_press=self.start_calibration
        )
        layout.add_widget(self.action_button)
        
        self.add_widget(layout)
        self.current_step = 0
        
    def start_calibration(self, *args):
        """Begins the three-step calibration process"""
        self.current_step = 1
        self.calibration_steps = [
            {
                'instruction': 'Point at the object and hold steady',
                'action': 'Capture Front',
                'angle': 0
            },
            {
                'instruction': 'Turn 90 degrees right',
                'action': 'Capture Side',
                'angle': 90
            },
            {
                'instruction': 'Level device horizontally',
                'action': 'Finish Calibration',
                'angle': None
            }
        ]
        self.update_calibration_ui()
        
    def update_calibration_ui(self):
        """Updates the UI for current calibration step"""
        if self.current_step <= len(self.calibration_steps):
            step = self.calibration_steps[self.current_step - 1]
            self.instruction_label.text = step['instruction']
            self.action_button.text = step['action']
            self.progress_label.text = f'Step {self.current_step}/3'
            
            # Add animation to make instructions more noticeable
            anim = Animation(opacity=0.5, duration=0.5) + Animation(opacity=1, duration=0.5)
            anim.repeat = True
            anim.start(self.instruction_label)
    
    def on_action_button(self, *args):
        """Handles calibration step completion"""
        if self.current_step <= len(self.calibration_steps):
            step = self.calibration_steps[self.current_step - 1]
            
            # Perform calibration action
            if step['angle'] is not None:
                self.pose_estimator.calibrate_angle(step['angle'])
            
            self.current_step += 1
            if self.current_step > len(self.calibration_steps):
                self.finish_calibration()
            else:
                self.update_calibration_ui()
    
    def finish_calibration(self):
        """Completes the calibration process"""
        self.instruction_label.text = 'Calibration Complete!'
        self.action_button.text = 'Start Capturing'
        Clock.schedule_once(self.dismiss, 2)