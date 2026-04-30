import numpy as np
from transformers import pipeline

class DepthEstimator:
    def __init__(self, model:str, device:str = "cuda"):
        self.pipe = pipeline(task="depth-estimation", model = model, device=device)

    def estimate(self, frame:np.ndarray) -> np.ndarray:
        pil_image = self.pipe(frame)["depth"] # return value of pipe is pil image
        return np.array(pil_image)