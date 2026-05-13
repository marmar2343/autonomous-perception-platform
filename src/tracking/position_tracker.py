import cv2
from src.fusion.types import Detection3D
import numpy as np
from collections import deque

class PositionTracker:
    def __init__(self):
        self.trajectories = {}

    def update(self,new_detections:list[Detection3D]):
        for detection in new_detections:
            if detection.track_id in self.trajectories:
                self.trajectories[detection.track_id].append(detection.bbox)
            
            else:
                self.trajectories[detection.track_id]=deque(maxlen=30)
                self.trajectories[detection.track_id].append(detection.bbox)
        
    def draw_trajectories(self,frame:np.ndarray) -> np.ndarray:
        for track_id in self.trajectories:
            points = list(self.trajectories[track_id])
            for i in range(len(points)-1):
                cx1 = int((points[i][0] + points[i][2]) / 2)
                cy1 = int((points[i][1] + points[i][3]) / 2)

                cx2 = int((points[i+1][0] + points[i+1][2]) / 2)
                cy2 = int((points[i+1][1] + points[i+1][3]) / 2)
                cv2.line(frame,(cx1,cy1),(cx2,cy2),(0,255,255),3)

        return frame
