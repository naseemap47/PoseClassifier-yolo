from ultralytics import YOLO
import cv2
import numpy as np
from utils import plot_skeleton_kpts





model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(2)
poses = None

while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Failed to Read...')
        break

    results = model.predict(img)
    for result in results:
        # print(result.keypoints)
        poses = result.keypoints

    # if poses is not None:
        for pose in poses:
            plot_skeleton_kpts(img, pose, radius=5, line_thick=2, confi=0.5)

    # img = cv2.resize(img, (1080, 720))
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
