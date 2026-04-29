import cv2
import sys
sys.path.append(".")

from src.detectors.detector import ObjectDetector



test_detector = ObjectDetector( "yolov8n.pt")
image = cv2.imread("scripts/test_image.jpg")

results = test_detector.detect(image)

for result in results:
    print(result.bbox, result.confidence)