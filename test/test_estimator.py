import numpy as np
from pose_estimators.feature_pose_estimator import FeaturePoseEstimator

def test_pose_estimator():
    estimator = FeaturePoseEstimator()
    # Create synthetic frame with known features
    frame = np.zeros((480, 640, 3))
    frame[100:150, 100:150] = 255  # Add white square
    
    pose = estimator.get_pose(frame)
    print(f"Estimated pose: {pose}")
