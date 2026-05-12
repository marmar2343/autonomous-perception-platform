import cv2
import sys
import time
import yaml
import numpy as np
import supervision as sv
sys.path.append(".")

from src.detectors.detector import ObjectDetector
from src.depth.depth_estimator import DepthEstimator
from src.fusion.fusion import fuse
from src.tracking.tracker import Tracker
from src.tracking.zone_detector import ZoneDetector

cap = cv2.VideoCapture("scripts/test_video.mp4")

with open("configs/zones.yaml") as f:
    config = yaml.safe_load(f)

test_detector = ObjectDetector( "yolov8n.pt")
test_estimator = DepthEstimator("depth-anything/Depth-Anything-V2-Small-hf")
test_tracker = Tracker()
test_zone_detector = ZoneDetector(np.array(config["zones"][0]["coordinates"]), config["zones"][0]["threshold"])
frame_count = 0



while True:
    start = time.time()

    ret, frame=cap.read()
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


    if not ret:
        print("End of video or error occured reading frame")
        break


    detections, result = test_detector.detect(frame)
    if(frame_count%3==0):
        depth_map = test_estimator.estimate(frame)

    tracks = test_tracker.track(result, test_detector.model.names)


    detection3D = fuse(tracks, depth_map)

    dangerous_zone_detection = test_zone_detector.detect_in_zone(detection3D)

    for result in detection3D:
        if result.track_id in dangerous_zone_detection:
            cv2.rectangle(frame,(int(result.bbox[0]), int(result.bbox[1])), (int(result.bbox[2]),int(result.bbox[3])), (0,0,255),3)
            frame = cv2.putText(img = frame, text = str(result.track_id) + " " + str(result.depth), org=(int(result.bbox[0]), int(result.bbox[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(0,0,255), thickness=3)
        else:
            cv2.rectangle(frame,(int(result.bbox[0]), int(result.bbox[1])), (int(result.bbox[2]),int(result.bbox[3])), (0,255,0),3)
            frame = cv2.putText(img = frame, text = str(result.track_id) + " " + str(result.depth), org=(int(result.bbox[0]), int(result.bbox[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(0,255,0), thickness=3)



    fps = 1/ (time.time()-start)
    print(f"FPS: {fps:.1f}")

    zone_points = np.array(config["zones"][0]["coordinates"])
    cv2.polylines(frame, [zone_points], isClosed=True, color=(0, 255, 255), thickness=5)
    
    # Display the frame
    frame = cv2.resize(frame, (480, 640))
    cv2.imshow("Video", frame)

    frame_count+=1

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

