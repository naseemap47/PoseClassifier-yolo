from keras.models import load_model
from ultralytics import YOLO
from utils import norm_kpts, plot_one_box, plot_skeleton_kpts, load_model_ext
import pandas as pd
import cv2
import time
import os
import argparse
import json


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to saved keras model")
ap.add_argument("-c", "--conf", type=float, default=0.25,
                help="path to saved keras model")
ap.add_argument("-s", "--source", type=str, required=True,
                help="path to video/cam/RTSP")
args = vars(ap.parse_args())

col_names = [
    '0_X', '0_Y', '1_X', '1_Y', '2_X', '2_Y', '3_X', '3_Y', '4_X', '4_Y', '5_X', '5_Y', 
    '6_X', '6_Y', '7_X', '7_Y', '8_X', '8_Y', '9_X', '9_Y', '10_X', '10_Y', '11_X', '11_Y', 
    '12_X', '12_Y', '13_X', '13_Y', '14_X', '14_Y', '15_X', '15_Y', '16_X', '16_Y'
]

# YOLOv8 Pose Model
model = YOLO('yolov8n-pose.pt')

# Keras pose model
saved_model, meta_str = load_model_ext(args['model'])
class_names = json.loads(meta_str)

# Load video/cam/RTSP
if args['source'].isnumeric():
    cap = cv2.VideoCapture(int(args['source']))
else:
    cap = cv2.VideoCapture(args['source'])

p_time = 0
while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Failed to Read...')
        break

    results = model.predict(img)
    for result in results:
        for box, pose in zip(result.boxes, result.keypoints):
            lm_list = []
            for pnt in pose:
                x, y = pnt[:2]
                lm_list.append([int(x), int(y)])
        
            if len(lm_list) == 17:
                pre_lm = norm_kpts(lm_list)
                data = pd.DataFrame([pre_lm], columns=col_names)
                predict = saved_model.predict(data)[0]

                if max(predict) > args['conf']:
                    pose_class = class_names[predict.argmax()]
                    # print('predictions: ', predict)
                    print('predicted Pose Class: ', pose_class)
                else:
                    pose_class = 'Unknown Pose'
                    print('[INFO] Predictions is below given Confidence!!')

                plot_one_box(box.xyxy[0], img, (255, 0, 255), f'{pose_class} {max(predict)}')
                plot_skeleton_kpts(img, pose, radius=5, line_thick=2, confi=0.5)

    # FPS
    c_time = time.time()
    fps = 1/(c_time-p_time)
    print('FPS: ', fps)
    p_time = c_time

    cv2.imshow('Output Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
