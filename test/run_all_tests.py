from test_estimator import test_pose_estimator
from test_detector import test_detector
from test_reconstructor import test_reconstructor   
if __name__ == "__main__":
    print("\nTesting Object Detector...")
    test_detector()

    print("Testing Pose Estimator...")
    test_pose_estimator()
    
    print("\nTesting Reconstructor...")
    test_reconstructor()
    