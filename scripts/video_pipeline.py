import cv2
import sys
sys.path.append(".")

from src.detectors.detector import ObjectDetector
from src.depth.depth_estimator import DepthEstimator
from src.fusion.fusion import fuse

cap = cv2.VideoCapture("scripts/test_video.mp4")

test_detector = ObjectDetector( "yolov8n.pt")
test_estimator = DepthEstimator("depth-anything/Depth-Anything-V2-Small-hf")

while True:
    ret, frame=cap.read()
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


    if not ret:
        print("End of video or error occured reading frame")
        break

    results = test_detector.detect(frame)
    depth_map = test_estimator.estimate(frame)

    detection3D = fuse(results, depth_map)

    for result in detection3D:
        cv2.rectangle(frame,(int(result.bbox[0]), int(result.bbox[1])), (int(result.bbox[2]),int(result.bbox[3])), (0,255,0),3)
        frame = cv2.putText(img = frame, text = str(result.depth), org=(int(result.bbox[0]), int(result.bbox[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(0,255,0), thickness=3)


    # Display the frame
    frame = cv2.resize(frame, (480, 640))
    cv2.imshow("Video", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break