import cv2
from src.fusion.types import Detection3D
import numpy as np
from collections import deque

class PositionTracker:
    def __init__(self):
        self.trajectories = {}

    # update trajectories with objects detected in current frame
    def update(self,new_detections:list[Detection3D]):
        # remove tracks for objects that left the frame
        current_ids = {d.track_id for d in new_detections}
        ids_to_remove = [id for id in self.trajectories if id not in current_ids]
        for id in ids_to_remove:
            del self.trajectories[id]
        
        # update trajectories, remember only last maxlen frames
        for detection in new_detections:
            if detection.track_id in self.trajectories:
                self.trajectories[detection.track_id].append(detection.bbox)
            
            else:
                self.trajectories[detection.track_id]=deque(maxlen=60)
                self.trajectories[detection.track_id].append(detection.bbox)
        
    # draw lines for objects that are currently in frame    
    def draw_trajectories(self,frame:np.ndarray) -> np.ndarray:
        for track_id in self.trajectories:
            points = list(self.trajectories[track_id])
            for i in range(len(points)-1):
                cx1 = int((points[i][0] + points[i][2]) / 2)
                cy1 = int((points[i][1] + points[i][3]) / 2)

                cx2 = int((points[i+1][0] + points[i+1][2]) / 2)
                cy2 = int((points[i+1][1] + points[i+1][3]) / 2)
                cv2.line(frame,(cx1,cy1),(cx2,cy2),(0,255,255),10)

        return frame
