from pose_estimators.feature_pose_estimator import FeaturePoseEstimator
from typing import List
import numpy as np
from core.base import ThreeDReconstructor

class ObjectReconstructor(ThreeDReconstructor):
   def __init__(self):
       self.points = []
       self.views = []
       self.pose_estimator = FeaturePoseEstimator()

   def reconstruct_object(self, frames: List[np.ndarray]) -> np.ndarray:
       for frame in frames:
           pose = self.pose_estimator.get_pose(frame)
           features = self._extract_features(frame)
           points_3d = self._triangulate_points(features, pose)
           self._merge_points(points_3d)
           
       return np.array(self.points)

   def _extract_features(self, frame):
       # Feature extraction code
       gray = np.mean(frame, axis=2)
       features = []
       for y in range(3, gray.shape[0]-3):
           for x in range(3, gray.shape[1]-3):
               window = gray[y-3:y+4, x-3:x+4]
               if self._is_feature(window):
                   features.append([x, y])
       return np.array(features)

   def _triangulate_points(self, features, pose):
       points_3d = []
       for x, y in features:
           depth = 1000  # Fixed depth for simplicity
           point = np.array([x, y, depth])
           R = self._euler_to_matrix(pose['azimuth'], pose['elevation'])
           world_point = np.dot(R, point)
           points_3d.append(world_point)
       return np.array(points_3d)

   def _merge_points(self, new_points):
       if not self.points:
           self.points = new_points
           return
           
       # Merge close points
       for point in new_points:
           distances = np.linalg.norm(self.points - point, axis=1)
           if np.min(distances) > 10:
               self.points.append(point)