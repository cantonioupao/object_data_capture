# tests/test_with_real_images.py
import cv2
import numpy as np
from pathlib import Path
from pose_estimators.feature_pose_estimator import FeaturePoseEstimator
from three_d_reconstructors.object_reconstructor import ObjectReconstructor
from object_detectors.simple_object_detection import ObjectDetector

def test_with_image_sequence():
   # Load test images
   image_dir = Path("test_images")
   images = []
   for img_path in image_dir.glob("*.jpg"):
       img = cv2.imread(str(img_path))
       if img is not None:
           images.append(img)

   # Test object detection
   detector = ObjectDetector()
   for i, img in enumerate(images):
       result = detector.detect_object(img)
       if result:
           # Draw detection results
           cv2.rectangle(img, 
                        (result['bounds']['x_min'], result['bounds']['y_min']),
                        (result['bounds']['x_max'], result['bounds']['y_max']), 
                        (0,255,0), 2)
           cv2.imwrite(f"detection_result_{i}.jpg", img)

   # Test pose estimation
   estimator = FeaturePoseEstimator()
   poses = []
   for img in images:
       pose = estimator.get_pose(img)
       poses.append(pose)
       print(f"Estimated pose: {pose}")

   # Test 3D reconstruction
   reconstructor = ObjectReconstructor()
   points_3d = reconstructor.reconstruct_object(images)
   
   # Save point cloud
   np.savetxt("reconstruction_result.xyz", points_3d)

if __name__ == "__main__":
   test_with_image_sequence()