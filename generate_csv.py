from ultralytics import YOLO
from utils import norm_kpts
import pandas as pd
import cv2
import os
import glob



path_data = 'yoga_poses/train'
path_to_save = 'data.csv'

col_names = []
for i in range(17):
    name_x = f'{i}_X'
    name_y = f'{i}_Y'
    col_names.append(name_x)
    col_names.append(name_y)

# YOLOv8 Pose Model
model = YOLO('yolov8n-pose.pt')

full_lm_list = []
target_list = []
class_names = sorted(os.listdir(path_data))
for class_name in class_names:
    path_to_class = os.path.join(path_data, class_name)
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
data.to_csv(path_to_save, encoding='utf-8', index=False)
print(f'[INFO] Successfully Saved Landmarks data into {path_to_save}')
