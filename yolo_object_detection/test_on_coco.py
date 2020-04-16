import os
import csv
import shutil
import time
import cv2
import numpy as np
import argparse
import yolo
import sys

# Compatible networks
NETWORKS = ["yolov3", "yolov3-tiny", "yolov3-tiny-prn"]
INPUT_SIZES = [320, 416, 608]

CONFIDENCE_THRESHOLD = 0.5
MNS_THRESHOLD = 0.4

images_path = "../test_images/coco_test/images/"

# List of all images for testing
images_list = os.listdir(images_path)

# Order images in ascending order
images_list = sorted(images_list, key=lambda x: int(x.replace(".jpg", "")))

#  Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--NN", default="yolov3", help="network to use; " + str(NETWORKS),
)
parser.add_argument(
    "-s", "--samples", default=None, type=int, help="number of images to test on",
)
parser.add_argument(
    "-g", "--gpu", default=False, type=bool, help="boolean to toggle the use of gpu"
)
parser.add_argument(
    "-i", "--input_size", default=608, type=int, help="Network resolution"
)
args = parser.parse_args()

# Default to all images
if args.samples is None or args.samples >= len(images_list):
    print(f"max number of samples is {len(images_list)}")
    args.samples = len(images_list)

# Load Yolo
try:
    model, classes, output_layers = yolo.load_network(args.NN, args.input_size)
except ValueError as err:
    # Invalid network parameters.
    print(str(err))
    sys.exit(1)

# Check for GPU
if args.gpu:
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print(f"\n[Loading] {args.NN}")
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
for n in range(args.samples):
    # Current image being processed
    print(f"Image:  ({counter}/{args.samples})   {images_list[n]}")

    # Execution time information
    frame_start_time = time.time()

    # Loading image
    img = cv2.imread(images_path + images_list[n])

    # Make result file for this image with the same name
    name = images_list[n].replace(".jpg", ".txt")
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
        layer_outputs, width, height, CONFIDENCE_THRESHOLD
    )

    # Non-maximal supresion to remove duplicate detections
    nms_indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, MNS_THRESHOLD
    )

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
    add_time_info([round(frame_end_time, 5), round(inf_time, 5), images_list[n]])

    # Increment image counter.
    counter += 1

# Time taken for code to finish
total_time = time.time() - start_time
add_time_info([total_time])

print(f"[Time taken] {total_time}")
