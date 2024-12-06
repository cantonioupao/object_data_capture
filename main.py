from core.capture_system import DamageCaptureSystem
from quality_analyzers.quality import BasicQualityAnalyzer
from pose_estimators.simple_pose_estimator import BasicPoseEstimator
from data_storage.sql_lite_manager import SQLiteStorageManager
from ui.app import DamageCaptureApp
from ui.advanced_features import AdvancedCaptureFeatures
import logging
import sys
from pathlib import Path

class CaptureApplication:
    """
    Main application coordinator that sets up and manages all components
    of the damage capture system.
    """
    def __init__(self):
        # Set up logging to track system operation
        self.setup_logging()
        
        # Initialize capture system components
        self.initialize_system()
        
        # Create and configure the UI application
        self.app = self.create_application()
        
        # Advanced features are initialized but not activated by default
        self.advanced_features = None
        
    def setup_logging(self):
        """Configures logging to help track system operation and debugging"""
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
        """Sets up core system components with proper configuration"""
        try:
            # Ensure storage directory exists
            Path("captures").mkdir(exist_ok=True)
            
            # Initialize system components
            self.quality_analyzer = BasicQualityAnalyzer()
            self.pose_estimator = BasicPoseEstimator()
            self.storage_manager = SQLiteStorageManager("captures.db")
            
            # Create main capture system
            self.capture_system = DamageCaptureSystem(
                quality_analyzer=self.quality_analyzer,
                pose_estimator=self.pose_estimator,
                storage_manager=self.storage_manager
            )
            
            self.logger.info("System components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            raise
            
    def create_application(self):
        """Creates and configures the UI application"""
        try:
            app = DamageCaptureApp(self.capture_system)
            self.logger.info("Application created successfully")
            return app
            
        except Exception as e:
            self.logger.error(f"Error creating application: {e}")
            raise
            
    def enable_advanced_features(self):
        """
        Activates advanced capture features when ready.
        This can be called after basic functionality is confirmed working.
        """
        try:
            if not self.advanced_features:
                self.advanced_features = AdvancedCaptureFeatures(self.app.root)
                self.advanced_features.activate()
                self.logger.info("Advanced features activated")
                
        except Exception as e:
            self.logger.error(f"Error enabling advanced features: {e}")
            
    def disable_advanced_features(self):
        """Deactivates advanced features if needed"""
        if self.advanced_features:
            self.advanced_features.deactivate()
            self.advanced_features = None
            self.logger.info("Advanced features deactivated")
            
    def run(self):
        """Starts the application"""
        try:
            self.logger.info("Starting application")
            self.app.run()
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Performs cleanup operations when application closes"""
        try:
            self.storage_manager.close()
            self.logger.info("Application cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

def main():
    """
    Application entry point that handles initialization and error management.
    """
    try:
        # Create and start the application
        application = CaptureApplication()
        
        # You can enable advanced features here when ready
        # application.enable_advanced_features()
        
        # Run the application
        application.run()
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()