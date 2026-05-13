import numpy as np
from src.fusion.types import Detection3D
import supervision as sv

class ZoneDetector:
    def __init__(self, polygon: np.ndarray, treshold: int = 30):
        self.polygon_zone = sv.PolygonZone(polygon=polygon) # define zone of interest with sv.PolygonZone
        self.treshold = treshold # no. of frames to determine that object is in dangerous zone
        self.track_id = {} # tracking ids of detected objects

    # find objects that are inn polygon_zone for too long
    def detect_in_zone(self, detections3D: list[Detection3D]) -> list[int]:
        # convert to sv.Detections format required by PolygonZone.trigger()
        xyxy = np.array([d.bbox for d in detections3D])
        tracker_ids = np.array([d.track_id for d in detections3D])
        sv_detections = sv.Detections(xyxy=xyxy, tracker_id=tracker_ids)
        is_detections_in_zone = self.polygon_zone.trigger(sv_detections)

        in_zone = [tracker_ids[i] for i in range(len(sv_detections)) if is_detections_in_zone[i]]

        for ids in tracker_ids:
            if ids in in_zone:
                self.track_id[ids] = self.track_id.get(ids, 0) + 1
            else:
                self.track_id[ids] = 0

        dangerous_zone = []
        for ids in self.track_id:
            if self.track_id[ids]>=self.treshold:
                dangerous_zone.append(ids)

        return dangerous_zone