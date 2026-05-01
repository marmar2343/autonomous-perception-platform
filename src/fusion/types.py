from src.detectors.types import Detection
from dataclasses import dataclass

@dataclass
class Detection3D(Detection):
    #Detection: bbox,class_name,confidence
    depth:float