import supervision as sv
from src.fusion.types import Detection3D
from ultralytics import Results


class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
    
    def track(self, result: Results, class_names: dict) -> list[Detection3D]:
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)

        detections3D = []

        for xyxy,_, confidence, class_id, tracker_id, _ in detections:

            detection3d = Detection3D(
            bbox=xyxy.tolist(),
            class_name=class_names[class_id],
            confidence=confidence,
            depth=-1,
            track_id = tracker_id)

            detections3D.append(detection3d)

        return detections3D

