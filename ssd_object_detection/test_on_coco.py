"""
This module provides code to run SSD Object Detection on the given folder
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
import argparse
import os
import csv
import shutil
import time

#  Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--conf", default=0.5, type=float, help="Set the detection threshold"
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
args = parser.parse_args()

# Compatible networks
MODEL = "MobileNetSSD_V2.pb"
INPUT_DIMENSIONS = 300

# Load class labels for relavant file.
# Based on https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

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

if args.samples is None or args.samples >= len(images_list):
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


# Load SSD
model = cv2.dnn.readNet(f"weights/{MODEL}", f"cfg/MobileNetSSD_V2.pbtxt")

# Check for GPU
if args.gpu:
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


print(f"\n[Data] {args.data}\n")
print(f"\n[Loading] {MODEL}")
print(f"[Input dimensions] {INPUT_DIMENSIONS} x {INPUT_DIMENSIONS}\n")

counter = 1

# Total execution time
start_time = time.time()

# Run inference on each image
for test_image in images_list:
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
    blob = cv2.dnn.blobFromImage(
        img, size=(INPUT_DIMENSIONS, INPUT_DIMENSIONS), swapRB=True, crop=False
    )

    # Run inference
    model.setInput(blob)
    start_inf = time.time()
    outs = model.forward()
    end_inf = time.time()

    # Network output information
    # Based on the code from:
    # https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
    # (accessed 12/01/20)
    for detection in outs[0, 0, :, :]:
        class_id = int(detection[1])
        confidence = float(detection[2])
        if confidence > args.conf:
            # Object detected
            x1 = detection[3] * width
            y1 = detection[4] * height
            x2 = detection[5] * width
            y2 = detection[6] * height

            class_name = str(classes[class_id])
            # remove spaces from class names for mAP score
            class_name = class_name.replace(" ", "", 1)
            line = f"{class_name} {confidence:.6f} {x1} {y1} {x2} {y2}"
            add_bb_info(detection_result_path, line)

    # Get inference time
    infer_time = end_inf - start_inf
    # print(infer_time)

    frame_end_time = time.time() - frame_start_time
    # Write times to file
    add_time_info([frame_end_time, infer_time])

    counter += 1


# Time taken for code to finish
total_time = time.time() - start_time
add_time_info([total_time])

print(f"[Time taken] {total_time}")
