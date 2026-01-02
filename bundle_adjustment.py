import numpy as np
import gtsam
from gtsam import Point3


class BundleAdjustment:
    def __init__(self, window_size=5, pose_std=1.0, landmark_std=1.0):
        self.window_size = window_size
    
        self.pose_noise = gtsam.noiseModel.Isotropic.Sigma(6, pose_noise)
        self.landmark_noise = gtsam.noiseModel.Isotropic.Sigma(3, landmark_noise)
        
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

    def add_pose(self, graph, initial_estimate, pose_id):
        pass
    
    def add_landmark(self, graph, initial_estimate, landmark_id, position):
        pass
    
    def add_projection_factor(self, graph, pose_id, landmark_id, measurement, K):
        pass
    
    def add_between_factor(self, graph, pose_id1, pose_id2, relative_pose):
        pass
    
    def marginalize_oldest_pose(self):
        pass
    
    def optimize(self):
        pass
        
