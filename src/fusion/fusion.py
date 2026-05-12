from .types import Detection3D
import numpy as np

def fuse(detections: list[Detection3D], depth_map: np.ndarray) ->  list[Detection3D]:
    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        depth = np.median(depth_map[y1:y2,x1:x2])

        detection.depth = depth


    return detections
