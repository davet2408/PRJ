import os
import csv
import shutil
import time
import cv2
import numpy as np
import argparse

# Compatible networks
NETWORKS = {
    "MobileNetSSD_V2.pb": "MobileNetSSD_V2.pbtxt",
    "SSD_512.caffemodel": "SSD_512.prototxt",
}
CONFIDENCE_THRESHOLD = 0.5
classes = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "unknown",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "unknown",
    "backpack",
    "umbrella",
    "unknown",
    "unknown",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "unknown",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "unknown",
    "dining table",
    "unknown",
    "unknown",
    "toilet",
    "unknown",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "unknown",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
# Load class labels for relavant file.
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

images_path = "../test_images/coco_test/images/"

nn_input_dimensions = 300

# List of all images for testing
images_list = os.listdir(images_path)

# Order images in ascending order
images_list = sorted(images_list, key=lambda x: int(x.replace(".jpg", "")))

#  Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--NN", default="MobileNetSSD_V2.pb", help="network to use; " + str(NETWORKS),
)
parser.add_argument(
    "-s", "--samples", default=None, type=int, help="number of images to test on",
)
parser.add_argument(
    "-g", "--gpu", default=False, type=bool, help="boolean to toggle the use of gpu"
)

args = parser.parse_args()

if args.NN not in NETWORKS:
    print(f"invalid NN '{args.NN}', chose from {str(NETWORKS)}")
    quit()

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
net = cv2.dnn.readNet(f"weights/{args.NN}", f"cfg/{NETWORKS[args.NN]}")

# Check for GPU
if args.gpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print(f"\n[Loading] {args.NN}")
print(f"[Input dimensions] {nn_input_dimensions} x {nn_input_dimensions}\n")

counter = 1

# Total execution time
start_time = time.time()

# Run inference on each image
for n in range(args.samples):
    print(f"Image:  ({counter}/{args.samples})   {images_list[n]}")

    # Execution time information
    frame_start_time = time.time()

    # Load image
    img = cv2.imread(images_path + images_list[n])

    # Make result file for this image with the same name
    name = images_list[n].replace(".jpg", ".txt")
    detection_result_path = os.path.join(detection_results, name)
    open(detection_result_path, "a")

    height, width, channels = img.shape

    # Format image
    blob = cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False)

    # Run inference
    net.setInput(blob)
    start_inf = time.time()
    outs = net.forward()
    end_inf = time.time()

    # Network output information
    for detection in outs[0, 0, :, :]:
        scores = detection[5:]
        class_id = int(detection[1])
        confidence = float(detection[2])
        if confidence > CONFIDENCE_THRESHOLD:
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
