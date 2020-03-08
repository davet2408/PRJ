import os
import sys
import cv2

image_dir = "test_images/visDrone/"

input_annotations_dir = image_dir + "VisDrone2019-DET-train/annotations"
input_images_dir = image_dir + "VisDrone2019-DET-train/images"

yolo_annotations_dir = image_dir + "VisDrone2019-DET-train-yolo/annotations"
voc_annotations_dir = image_dir + "VisDrone2019-DET-train-voc/annotations"

os.makedirs(yolo_annotations_dir, exist_ok=True)
os.makedirs(voc_annotations_dir, exist_ok=True)

class_label = {
    "0": "Ignore",
    "1": "Pedestrian",
    "2": "People",
    "3": "Bicycle",
    "4": "Car",
    "5": "Van",
    "6": "Truck",
    "7": "Tricycle",
    "8": "Awning-tricycle",
    "9": "Bus",
    "10": "Motor",
    "11": "Others",
}

annotation_list = os.listdir(input_annotations_dir)
images_list = os.listdir(input_images_dir)

for annotation in annotation_list:

    annotation_path = os.path.join(os.getcwd(), input_annotations_dir, annotation)

    image_file = annotation.split(".txt")[0] + ".jpg"
    image_path = os.path.join(os.getcwd(), input_images_dir, image_file)
    img = cv2.imread(image_path)

    w = img.shape[1]
    h = img.shape[2]

    with open(annotation_path, "r") as file:
        lines = file.readlines()

        voc_lines = []
        yolo_lines = []

        bbs = []

        for line in lines:
            line = line.strip("\n").split(",")
            x1 = int(line[0])
            y1 = int(line[1])
            bb_w = int(line[2])
            bb_h = int(line[3])
            x2 = int(line[0]) + w
            y2 = int(line[1]) + h
            mid_x = x2 / 2.0
            mid_y = y2 / 2.0

            label = class_label.get(line[5])
            if line[5] == "0":
                continue
            voc_lines.append(f"{label} {x1} {y1} {x2} {y2}")
            yolo_lines.append(f"{label} {mid_x/w} {mid_y/h} {bb_w/w} {bb_h/h}")
