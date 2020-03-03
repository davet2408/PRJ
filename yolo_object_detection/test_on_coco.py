import os
import csv
import shutil
import time
import cv2
import numpy as np
import argparse

# Compatible networks
NETWORKS = ["yolov3", "yolov3-tiny", "yolov3-tiny-prn"]
CONFIDENCE_THRESHOLD = 0.25

# images_path = "test_images/coco_val_full/images/"
images_path = "test_images/coco_test/images/"

nn_input_dimensions = 416

# List of all images for testing
# images_list = os.listdir("test_images/coco_test/images")
images_list = os.listdir(images_path)

#  Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--NN",
    default="yolov3",
    help="Chose the NN to use, current options yolov3 and yolov3-tiny",
)
parser.add_argument(
    "-s",
    "--samples",
    default=len(images_list),
    type=int,
    help="specify the number of images to test on, limit is 500",
)

args = parser.parse_args()

if args.NN not in NETWORKS:
    print(f"invalid NN '{args.NN}', chose from {str(NETWORKS)}")
    quit()

if args.samples >= len(images_list):
    print(f"max number of samples is {len(images_list)}")
    args.samples = len(images_list)

# Directories for results
detection_results = "results/detection-results"
time_results = "results/time-results.csv"

# Remove any exisiting results files to prevent overlap
if os.path.exists(detection_results):
    shutil.rmtree(detection_results)
if os.path.exists(time_results):
    os.remove(time_results)

os.makedirs(detection_results)


def add_bb_info(path, line):
    # Store bounding box information for mAP score
    with open(path, "a") as file:
        file.write(line + "\n")


def add_time_info(times):
    # Store execution time information
    times = [round(time, 5) for time in times]
    with open(time_results, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(times)


# Load Yolo
net = cv2.dnn.readNet(f"weights/{args.NN}.weights", f"cfg/{args.NN}.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

counter = 1

# Run inference on each image
for n in range(args.samples):
    print(f"Image:  ({counter}/{args.samples})   {images_list[n]}")

    # Execution time information
    start_time = time.time()

    # Loading image
    img = cv2.imread(images_path + images_list[n])

    # Make result file for this image with the same name
    name = images_list[n].replace(".jpg", ".txt")
    detection_result_path = os.path.join(detection_results, name)
    open(detection_result_path, "a")

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        img,
        0.00392,
        (nn_input_dimensions, nn_input_dimensions),
        (0, 0, 0),
        True,
        crop=False,
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Network output information
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximal supresion to remove duplicate detections
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)

    # font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            class_name = str(classes[class_ids[i]])
            # remove spaces from class names for mAP score
            class_name = class_name.replace(" ", "", 1)
            # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            line = f"{class_name} {confidences[i]:.6f} {x} {y} {x+w} {y+h}"
            add_bb_info(detection_result_path, line)

    # Get inference time
    t, _ = net.getPerfProfile()
    infer_time = t / cv2.getTickFrequency()
    # infer_time = f"Inference time: {t / cv2.getTickFrequency()}"
    # print(infer_time)

    total_time = time.time() - start_time
    # Write times to file
    add_time_info([total_time, infer_time])

    counter += 1

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()

