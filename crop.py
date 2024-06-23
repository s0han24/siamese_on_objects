import os
import cv2
from ultralytics import YOLO

path_to_model = 'models\\object detection\\best.pt' # Path to the trained YOLOv8 model
path_to_images_folder = 'images2' # Path to the folder containing the images
image_file = '218_QS2551564_AddVisibilityImg-20240417_170252.jpg' # Image file name
crop_dir_name = "crop_images" # Name of the folder to save the cropped images

model = YOLO(path_to_model)
names = model.names

if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)

idx = 0
im0 = cv2.imread(os.path.join(path_to_images_folder, image_file))
results = model.predict(im0, show=False)
boxes = results[0].boxes.xyxy.cpu().tolist()
clss = results[0].boxes.cls.cpu().tolist()

if boxes is not None:
    for box, cls in zip(boxes, clss):
        idx += 1

        crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

        cv2.imwrite(os.path.join(crop_dir_name,  str(idx) + ".png"), crop_obj)