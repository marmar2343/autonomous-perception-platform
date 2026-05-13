import supervision as sv
from src.fusion.types import Detection3D
from ultralytics.engine.results import Results


class Tracker:
    def __init__(self):
        # lost_track_buffer: number of frames to keep a track alive after object disappears
        self.tracker = sv.ByteTrack(lost_track_buffer=60)
    
    # track all objects that YOLO detected and update tracker with update_with_detection, assign persistent IDs, return list of Detection3D
    def track(self, result: Results, class_names: dict) -> list[Detection3D]:
        # convert YOLO result to supervision format for ByteTrack
        detections = sv.Detections.from_ultralytics(result)
        # update tracker state and assign persistent track IDs    
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

