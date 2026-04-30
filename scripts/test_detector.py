import cv2
import sys
sys.path.append(".")

from src.detectors.detector import ObjectDetector
from src.depth.depth_estimator import DepthEstimator



test_detector = ObjectDetector( "yolov8n.pt")
test_estimator = DepthEstimator("depth-anything/Depth-Anything-V2-Small-hf")
image = cv2.imread("scripts/test_image.jpg")
image_detected = image.copy()

results = test_detector.detect(image_detected)

for result in results:
    cv2.rectangle(image_detected,(int(result.bbox[0]), int(result.bbox[1])), (int(result.bbox[2]),int(result.bbox[3])), (0,255,0),3)
    image_detected = cv2.putText(img = image_detected, text = result.class_name, org=(int(result.bbox[0]), int(result.bbox[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=3)

image_depth = test_estimator.estimate(image)

cv2.imshow('Title', image_detected)
cv2.imshow('Depth map',image_depth)
cv2.waitKey(0)