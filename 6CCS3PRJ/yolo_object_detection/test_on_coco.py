"""
This module provides code to run yolo Object Detection on the given folder
of images and save the prediction in voc format to results/detection-results.
Timing information for both code execution and inference time is also 
recorded to a CSV file "results/time-results.csv".

detection-results can then be passed to the mapscore library to generate a
mAP score for these detections against the ground truth values for the data.


Author: David Temple
Date: 02/03/2020
"""
# OpenCV module installed from https://github.com/opencv/opencv
import cv2

# numpy module installed via pip https://numpy.org
import numpy as np

# Python standard library modules
import os
import csv
import shutil
import time
import argparse
import sys

# Hand made modules
import yolo


#  Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="yolov3",
    help="network to use: " + str(yolo.NETWORKS.keys()),
)
parser.add_argument(
    "-c", "--conf", default=0.5, type=float, help="Set the detection threshold"
)
parser.add_argument(
    "--NMS", default=0.4, type=float, help="Non Maximum Supression threshold"
)
parser.add_argument(
    "-d",
    "--data",
    default="../test_images/coco_test/images/",
    help="relative path of dataset to use",
)
parser.add_argument(
    "-s", "--samples", default=None, type=int, help="number of images to test on",
)
parser.add_argument(
    "-g", "--gpu", default=False, type=bool, help="boolean to toggle the use of gpu"
)
parser.add_argument(
    "-i",
    "--input_size",
    default=608,
    type=int,
    help="Network resolutions: " + str(yolo.INPUT_SIZES),
)
args = parser.parse_args()

# Select dataset
images_path = args.data if args.data[-1] == "/" else args.data + "/"
# List of all images for testing
images_list = os.listdir(images_path)
# Order images in ascending order if possible
images_list = sorted(
    images_list,
    key=lambda x: int(x.replace(".jpg", ""))
    if x.replace(".jpg", "").isnumeric()
    else 1,
)

# Default to all images
if args.samples is None or args.samples >= len(images_list):
    print(f"max number of samples is {len(images_list)}")
    args.samples = len(images_list)

# Load Yolo
try:
    model, classes, output_layers = yolo.load_network(args.model, args.input_size)
except ValueError as err:
    # Invalid network parameters.
    print(str(err))
    sys.exit(1)

# Check for GPU
if args.gpu:
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print(f"\n[Data] {args.data}\n")
print(f"\n[Loading] {args.model}")
print(f"[Input dimensions] {args.input_size} x {args.input_size}\n")

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
    with open(time_results, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(times)


# Track current image
counter = 1

# Total execution time
start_time = time.time()

# Run inference on each image
for test_image in images_list:
    # Current image being processed
    print(f"Image:  ({counter}/{args.samples})   {test_image}")

    # Execution time information
    frame_start_time = time.time()

    # Load image
    img = cv2.imread(images_path + test_image)

    # Make result file for this image with the same name
    name = test_image.replace(".jpg", ".txt")
    detection_result_path = os.path.join(detection_results, name)
    open(detection_result_path, "a")

    height, width, channels = img.shape

    # Format image
    blob = yolo.get_blob(img, args.input_size)

    # Run inference
    model.setInput(blob)
    start_inf = time.time()
    layer_outputs = model.forward(output_layers)
    end_inf = time.time()

    # Network prediction information
    class_ids, confidences, boxes = yolo.get_detections(
        layer_outputs, width, height, args.conf
    )

    # Non-maximal supresion to remove duplicate detections
    nms_indexes = cv2.dnn.NMSBoxes(boxes, confidences, args.conf, args.NMS)

    # Get final prediction information.
    # Based on code from:
    #    https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
    #    (accessed 12/01/20)
    for box_idx in range(len(boxes)):
        if box_idx in nms_indexes:
            x, y, w, h = boxes[box_idx]
            label = str(classes[class_ids[box_idx]])
            class_name = str(classes[class_ids[box_idx]])
            # Remove spaces from class names for mAP score.
            class_name = class_name.replace(" ", "", 1)
            line = f"{class_name} {confidences[box_idx]:.6f} {x} {y} {x+w} {y+h}"
            # Write predictions to file
            add_bb_info(detection_result_path, line)

    # Get inference time
    inf_time = end_inf - start_inf
    # print(infer_time)

    frame_end_time = time.time() - frame_start_time
    # Write times to file
    add_time_info([round(frame_end_time, 5), round(inf_time, 5), test_image])

    # Increment image counter.
    counter += 1

# Time taken for code to finish
total_time = time.time() - start_time
add_time_info([total_time])

print(f"[Time taken] {total_time}")
