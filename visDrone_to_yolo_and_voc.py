import os
import sys
import cv2
import numpy as np

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
    "2": "Pedestrian",  #  origonally labeled as people but this seems superflous so changed to pedestrian aswell
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

# ordered in; Pedestrian, Bicycle, Car, Van ,Truck, Tricycle, Awning-tricycle, Bus, Motor, Others
class_id = {
    "0": "",
    "1": "0",
    "2": "0",  #  origonally labeled as people but this seems superflous so changed to pedestrian aswell
    "3": "1",
    "4": "2",
    "5": "3",
    "6": "4",
    "7": "5",
    "8": "6",
    "9": "7",
    "10": "8",
    "11": "9",
}

colors = np.random.uniform(0, 255, size=(len(class_label), 3))

annotation_list = os.listdir(input_annotations_dir)
images_list = os.listdir(input_images_dir)

count = 0

for annotation in annotation_list:

    if count == 10:
        break

    annotation_path = os.path.join(os.getcwd(), input_annotations_dir, annotation)

    image_file = annotation.split(".txt")[0] + ".jpg"
    image_path = os.path.join(os.getcwd(), input_images_dir, image_file)
    img = cv2.imread(image_path)

    width = img.shape[0]
    height = img.shape[1]
    dw = 1.0 / img.shape[0]
    dh = 1.0 / img.shape[1]

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
            label = class_label.get(line[5])

            if line[5] == "0":
                # this is the label for ignore so we skip it
                skips += 1
                continue
            voc_lines.append(f"{label} {x1} {y1} {x2} {y2}")
            yolo_lines.append(f"{label} {mid_x*dw} {mid_y*dh} {bb_w*dw} {bb_h*dh}")

            # voc style
            # bbs.append([line[5], x1, x2, y1, y2])

            # yolo style
            bbs.append([line[5], mid_x * dw, mid_y * dh, bb_w * dw, bb_h * dh])

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        for i in range(len(bbs)):
            # from voc format
            # class_i, x1, x2, y1, y2 = bbs[i]

            # from yolo
            class_i = bbs[i][0]
            center_x = int(bbs[i][1] * width)
            center_y = int(bbs[i][2] * height)
            w = int(bbs[i][3] * width)
            h = int(bbs[i][4] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            color = colors[int(class_i)]
            label = class_label[class_i]

            # voc
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(img, label, (x1, y1), font, 3, color, 2)

            # yolo
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if skips == len(lines):
            # All annotations are Nothing so entire file skipped
            # Print the filename if it is skipped so it can be checked manualy
            print(f"================{annotation}================")

        # print(yolo_lines)
        # print(len(voc_lines))

        count += 1
        print(f"{count} / {len(annotation_list)}")
