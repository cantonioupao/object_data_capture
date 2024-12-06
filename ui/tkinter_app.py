from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
from visualization.basic_visualizer import BasicVisualizer
import threading
from queue import Queue
import time
from core.config import CaptureConfig

class CaptureWindow:
    def __init__(self, capture_system, root):
        self.root = root
        self.root.title('3D Object Capture System')
        self.width, self.height = CaptureConfig.CAMERA_RESOLUTION
        self.root.geometry(f'{self.width+320}x{self.height+180}')
        
        self.capture_system = capture_system
        self.visualizer = BasicVisualizer(self.width, self.height)
        self.photo = None
        self.frame_queue = Queue(maxsize=1)  # Reduced queue size
        self.running = True
        self.frame_interval = 1.0 / CaptureConfig.CAMERA_FPS
        
        self.init_ui()
        self.setup_camera()

    def init_ui(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True)
        
        self.camera_label = ttk.Label(main_frame)
        self.camera_label.pack(pady=10)
        
        self.status_label = ttk.Label(main_frame, text='Ready')
        self.status_label.pack(pady=5)
        
        self.capture_button = ttk.Button(main_frame, text='Capture', command=self.capture_image)
        self.capture_button.pack(pady=5)

    def setup_camera(self):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, CaptureConfig.CAMERA_FPS)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        threading.Thread(target=self.capture_loop, daemon=True).start()
        self.root.after(int(self.frame_interval * 1000), self.update_frame)

    def capture_loop(self):
        last_frame_time = time.time()
        
        while self.running:
            if time.time() - last_frame_time >= self.frame_interval:
                ret, frame = self.camera.read()
                if ret and not self.frame_queue.full():
                    result = self.capture_system.process_frame(frame)
                    frame = self.visualizer.draw_capture_zones(frame, self.capture_system.capture_zones)
                    
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                            
                    self.frame_queue.put((frame, result['message']))
                    last_frame_time = time.time()
            time.sleep(self.frame_interval / 2)

    def update_frame(self):
        try:
            frame, message = self.frame_queue.get_nowait()
            self.status_label.config(text=message)
            
            if frame is not None:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                
                if self.photo is None:
                    self.photo = ImageTk.PhotoImage(image=image)
                else:
                    self.photo.paste(image)
                
                self.camera_label.configure(image=self.photo)
        except:
            pass
            
        self.root.after(int(self.frame_interval * 1000), self.update_frame)

    def capture_image(self):
        ret, frame = self.camera.read()
        if ret:
            success = self.capture_system.capture(frame)
            self.status_label.config(text="Success" if success else "Failed")

    def on_closing(self):
        self.running = False
        if hasattr(self, 'camera'):
            self.camera.release()
        self.root.destroy()

def create_ui(capture_system, root):
    return CaptureWindow(capture_system, root)