from ultralytics import YOLO
from utils import norm_kpts
import pandas as pd
import cv2
import os
import glob
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--data", type=str, required=True,
                help="path to data/dir")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save csv file, eg: dir/data.csv")
args = vars(ap.parse_args())

col_names = [
    '0_X', '0_Y', '1_X', '1_Y', '2_X', '2_Y', '3_X', '3_Y', '4_X', '4_Y', '5_X', '5_Y', 
    '6_X', '6_Y', '7_X', '7_Y', '8_X', '8_Y', '9_X', '9_Y', '10_X', '10_Y', '11_X', '11_Y', 
    '12_X', '12_Y', '13_X', '13_Y', '14_X', '14_Y', '15_X', '15_Y', '16_X', '16_Y'
]

# YOLOv8 Pose Model
model = YOLO('yolov8n-pose.pt')

full_lm_list = []
target_list = []
class_names = sorted(os.listdir(args['data']))
for class_name in class_names:
    path_to_class = os.path.join(args['data'], class_name)
    img_list = glob.glob(path_to_class + '/*.jpg') + \
        glob.glob(path_to_class + '/*.jpeg') + \
        glob.glob(path_to_class + '/*.png')
    img_list = sorted(img_list)

    for img_path in img_list:
        img = cv2.imread(img_path)
        lm_list = []
        if img is None:
            print(
                f'[ERROR] Error in reading {img_path} -- Skipping.....\n[INFO] Taking next Image')
            continue
        else:
            results = model.predict(img)
            for result in results:
                poses = result.keypoints
                for pose in poses:
                    for pnt in pose:
                        x, y = pnt[:2]
                        lm_list.append([int(x), int(y)])
        
        if len(lm_list) == 17:
            pre_lm = norm_kpts(lm_list)
            full_lm_list.append(pre_lm)
            target_list.append(class_name)

        print(f'{os.path.split(img_path)[1]} Landmarks added Successfully')
    print(f'[INFO] {class_name} Successfully Completed')
print('[INFO] Landmarks from Dataset Successfully Completed')

# to csv
data_x = pd.DataFrame(full_lm_list, columns=col_names)
data = data_x.assign(Pose_Class=target_list)
data.to_csv(args['save'], encoding='utf-8', index=False)
print(f"[INFO] Successfully Saved Landmarks data into {args['save']}")
