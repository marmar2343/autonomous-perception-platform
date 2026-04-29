import cv2
import sys
sys.path.append(".")

from src.detectors.detector import ObjectDetector



test_detector = ObjectDetector( "yolov8n.pt")
image = cv2.imread("scripts/test_image.jpg")

results = test_detector.detect(image)

for result in results:
    cv2.rectangle(image,(int(result.bbox[0]), int(result.bbox[1])), (int(result.bbox[2]),int(result.bbox[3])), (0,255,0),3)
    image = cv2.putText(img = image, text = result.class_name, org=(int(result.bbox[0]), int(result.bbox[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=3)

cv2.imshow('Title', image)
cv2.waitKey(0)