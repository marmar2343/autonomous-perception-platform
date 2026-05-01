from .types import Detection3D
import numpy as np
from ..detectors.types import Detection

def fuse(detections: list[Detection], depth_map: np.ndarray) ->  list[Detection3D]:
    detections_fuse = []
    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        depth = np.median(depth_map[y1:y2,x1:x2])

        detection3d = Detection3D(
        bbox=detection.bbox,
        class_name=detection.class_name,
        confidence=detection.confidence,
        depth=depth)

        detections_fuse.append(detection3d)

    return detections_fuse
