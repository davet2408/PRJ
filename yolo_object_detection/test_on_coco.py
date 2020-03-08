import os
import csv
import shutil
import time
import cv2
import numpy as np
import argparse

# Compatible networks
NETWORKS = ["yolov3", "yolov3-tiny", "yolov3-tiny-prn"]
INPUT_SIZES = [320, 416, 608]
CONFIDENCE_THRESHOLD = 0.5

# images_path = "../test_images/coco_val_full/images/"
images_path = "../test_images/coco_test/images/"

# List of all images for testing
images_list = os.listdir(images_path)

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
    "-i", "--input_size", default=608, type=int, help="Network input size"
)

args = parser.parse_args()

if args.NN not in NETWORKS:
    print(f"invalid NN '{args.NN}', chose from {str(NETWORKS)}")
    quit()

if args.samples is None or args.samples >= len(images_list):
    print(f"max number of samples is {len(images_list)}")
    args.samples = len(images_list)

if args.input_size in INPUT_SIZES:
    nn_input_dimensions = args.input_size
else:
    print(f"Valid network input sizes: {str(INPUT_SIZES)}")

# Load Yolo
net = cv2.dnn.readNet(f"weights/{args.NN}.weights", f"cfg/{args.NN}.cfg")

# Check for GPU
if args.gpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

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
    times = [round(time, 5) for time in times]
    with open(time_results, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(times)


classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

counter = 1

# Total execution time
start_time = time.time()

# Run inference on each image
for n in range(args.samples):
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

    frame_end_time = time.time() - frame_start_time
    # Write times to file
    add_time_info([frame_end_time, infer_time])

    counter += 1

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

# Time taken for code to finish
total_time = time.time() - start_time
add_time_info([total_time])

print(f"[Time taken] {total_time}")

# cv2.destroyAllWindows()

