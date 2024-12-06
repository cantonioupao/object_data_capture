from core.capture_system import DamageCaptureSystem
from object_detectors.simple_object_detection import SimpleObjectDetector
from quality_analyzers.quality import BasicQualityAnalyzer
from pose_estimators.feature_pose_estimator import FeaturePoseEstimator
from data_storage.sql_lite_manager import SQLiteStorageManager
from ui.tkinter_app import create_ui
import tkinter as tk
import logging
import sys
from pathlib import Path
from core.config import CaptureConfig

class CaptureApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_logging()
        self.initialize_system()
        self.window = create_ui(self.capture_system, self.root)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('capture_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_system(self):
        try:
            Path("captures").mkdir(exist_ok=True)
            
            self.object_detector = SimpleObjectDetector()
            self.quality_analyzer = BasicQualityAnalyzer()
            self.pose_estimator = FeaturePoseEstimator()
            self.storage_manager = SQLiteStorageManager("captures.db")
            
            self.capture_system = DamageCaptureSystem(
                object_detector=self.object_detector,
                quality_analyzer=self.quality_analyzer,
                pose_estimator=self.pose_estimator,
                storage_manager=self.storage_manager,
                config=CaptureConfig
            )
            
            self.logger.info("System components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            raise

    def run(self):
        try:
            self.logger.info("Starting application")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            if hasattr(self, 'window') and hasattr(self.window, 'camera'):
                self.window.camera.release()
            self.storage_manager.close()
            self.logger.info("Application cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

def main():
    try:
        app = CaptureApplication()
        app.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()