from .types import Detection
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, path:str):
        self.model = YOLO(path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                detection = Detection(box.xyxy.tolist()[0],self.model.names[box.cls.item()],box.conf.item())
                detections.append(detection)
        return detections

