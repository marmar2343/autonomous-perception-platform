from .types import Detection
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, path:str):
        self.model = YOLO(path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        pass

