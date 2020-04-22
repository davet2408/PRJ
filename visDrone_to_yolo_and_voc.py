"""
Script to convert the VisDrone systle data into YOLO and VOC format.

Author: David Temple
Date: 02/02/2020
"""
# Python standard library modules
import os
import sys
import cv2

# numpy module installed via pip https://numpy.org
import numpy as np


def add_line(path, line):
    # write annotation information to new file
    with open(path, "a") as file:
        file.write(line + "\n")


# Class labels for retraining v1
# class_label = {
#     "0": ("Ignore", ""),
#     "1": ("Pedestrian", "0"),
#     #  origonally labeled as people but this seems superflous so changed to pedestrian aswell
#     "2": ("Pedestrian", "0",),
#     "3": ("Bicycle", "1"),
#     "4": ("Car", "2"),
#     "5": ("Van", "3"),
#     "6": ("Truck", "4"),
#     "7": ("Tricycle", "5"),
#     "8": ("Awning-tricycle", "6"),
#     "9": ("Bus", "7"),
#     "10": ("Motor", "8"),
#     "11": ("Others", "9"),
# }

# Class labels for retraining v2
class_label = {
    "0": ("Ignore", ""),
    "1": ("Pedestrian", "0"),
    #  origonally labeled as people but this seems superflous so changed to pedestrian aswell
    "2": ("Pedestrian", "0",),
    "3": ("Bike", "1"),
    "4": ("Car", "2"),
    "5": ("Van", "3"),
    "6": ("Truck", "4"),
    "7": ("Tricycle", "5"),
    "8": ("Tricycle", "5"),
    "9": ("Bus", "6"),
    "10": ("Bike", "1"),
    "11": ("Others", ""),
}

ignore = ["0", "11"]

colors = np.random.uniform(0, 255, size=(len(class_label), 3))

image_dir = "test_images/visDrone/"

train_path = "data/obj/"

obj = "test_images/visDrone/obj/"

try:
    os.makedirs(obj, exist_ok=False)
except FileExistsError:
    print("obj file already exisits, move it and run again")
    quit()

total_len = 0

for task in ["train", "test-dev", "val"]:

    input_annotations_dir = image_dir + f"VisDrone2019-DET-{task}/annotations"
    input_images_dir = image_dir + f"VisDrone2019-DET-{task}/images"

    yolo_annotations_dir = image_dir + f"VisDrone-{task}-yolo/"
    voc_annotations_dir = image_dir + f"VisDrone-{task}-voc/"

    os.makedirs(yolo_annotations_dir, exist_ok=True)
    os.makedirs(voc_annotations_dir, exist_ok=True)

    annotation_list = os.listdir(input_annotations_dir)
    images_list = os.listdir(input_images_dir)

    print(len(annotation_list))

    total_len += len(annotation_list) * 2

    count = 0

    for annotation in annotation_list:

        # if count == 5:
        #     break

        annotation_path = os.path.join(os.getcwd(), input_annotations_dir, annotation)

        image_file = annotation.split(".txt")[0] + ".jpg"
        image_path = os.path.join(os.getcwd(), input_images_dir, image_file)
        img = cv2.imread(image_path)

        height = img.shape[0]
        width = img.shape[1]
        dh = 1.0 / height
        dw = 1.0 / width

        add_line(f"{image_dir}{task}.txt", train_path + image_file)

        if task != "test-dev":
            cv2.imwrite(obj + image_file, img)

        # continue

        with open(annotation_path, "r") as file:
            lines = file.readlines()

            skips = 0

            voc_lines = []
            yolo_lines = []

            bbs = []

            for line in lines:
                line = line.strip("\n").split(",")

                # bounding box width and height
                bb_w = int(line[2])
                bb_h = int(line[3])
                # top left x,y
                x1 = int(line[0])
                y1 = int(line[1])
                # bottom right x,y
                x2 = int(line[0]) + bb_w
                y2 = int(line[1]) + bb_h
                # middle of bounding box x,y
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0
                # class label for bounding box
                label = class_label.get(line[5])[0]
                class_id = class_label.get(line[5])[1]

                if line[5] in ignore:
                    # this is the label for ignore so we skip it
                    skips += 1
                    continue

                voc_line = f"{label} {x1} {y1} {x2} {y2}"
                yolo_line = f"{class_id} {mid_x*dw} {mid_y*dh} {bb_w*dw} {bb_h*dh}"

                if mid_x * dw > 1 or mid_y * dh > 1 or mid_x * dw < 0 or mid_y * dh < 0:
                    print("wrong")

                # print(yolo_annotations_dir + annotation, yolo_line)

                if task != "test-dev":
                    add_line(obj + annotation, yolo_line)
                add_line(yolo_annotations_dir + annotation, yolo_line)
                add_line(voc_annotations_dir + annotation, voc_line)

            if skips == len(lines):
                # All annotations are ignored so entire file skipped
                # Print the filename if it is skipped so it can be checked manualy
                print(f"================{annotation}================")

            # print(yolo_lines)
            # print(len(voc_lines))

            count += 1
            print(f"{count} / {len(annotation_list)}")

# print("total ==", total_len)
# print("obj == ", len(os.listdir(obj)))
