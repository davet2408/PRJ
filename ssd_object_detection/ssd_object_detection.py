"""
This program applies Object Detection to the video or images provided to it
allowing the testing of SSD object detectors. Parameters can be altered to 
try and get the best possible performance. 

There are a limited number of ways to interpret the SSD output from OpenCV
therefore, these sources have been considered and used in the development of 
this file.

Sections of this code is based on implimentations from these sources:
https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
(accessed 12/01/20)

https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-
and-opencv/
(accessed 20/01/20)

https://medium.com/@franky07724_57962/exploring-opencvs-deep-learning-object-
detection-library-e51fe7c82246
(accessed 12/01/20)

Author: David Temple
Date: 02/03/2020
"""
# OpenCV module installed from https://github.com/opencv/opencv
import cv2

# numpy module installed via pip https://numpy.org
import numpy as np

# Python standard library modules
import argparse
import sys
import time

#  Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", default=None, help="Relative path to video file to run yolo on."
)
parser.add_argument(
    "--image",
    default="../test_images/coco_test/images/1.jpg",
    help="Relative path to image file to run SSD on.",
)
parser.add_argument(
    "-c", "--conf", default=0.5, type=float, help="Set the detection threshold"
)
parser.add_argument(
    "-g", "--gpu", default=False, type=bool, help="boolean to toggle the use of gpu"
)
parser.add_argument(
    "-t", "--text", default=True, type=bool, help="Show prediction text"
)
args = parser.parse_args()


INPUT_DIMENSIONS = 300
MODEL = "MobileNetSSD_V2"


def run_infernce(frame, args, model, classes):
    """Runs Object Detection on the given frame. Adds bounding box predictions
    to the frame with the class colours provided stating confidence score.
    Low confidence predictions are removed.

    Function based on the code from:
    https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
    (accessed 12/01/20)
    """

    height, width, channels = frame.shape

    # Format image
    blob = cv2.dnn.blobFromImage(
        frame, size=(INPUT_DIMENSIONS, INPUT_DIMENSIONS), swapRB=True, crop=False
    )

    # Run inference
    model.setInput(blob)
    start_inf = time.time()
    outs = model.forward()
    end_inf = time.time()

    # Network output information
    for detection in outs[0, 0, :, :]:
        class_id = int(detection[1])
        confidence = float(detection[2])
        if confidence > args.conf:
            # Object detected
            x1 = detection[3] * width
            y1 = detection[4] * height
            x2 = detection[5] * width
            y2 = detection[6] * height

            label = str(classes[class_id])
            colour = colours[class_id]
            # Draw bounding boxes
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (int(x1), int(y1) - 3),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                colour,
                thickness=2,
            )


# Load class labels for relavant file.
# Based on https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colours = np.random.uniform(0, 255, size=(len(classes), 3))

model = cv2.dnn.readNet(f"weights/{MODEL}.pb", f"cfg/{MODEL}.pbtxt")


# Video detection.
if args.video:
    # Webcam
    if args.video == "0":
        cap = cv2.VideoCapture(0)
    # Video file
    else:
        cap = cv2.VideoCapture(args.video)

    while True:

        ret, frame = cap.read()
        if not ret:
            print(f"No video found at {args.video}")
            sys.exit(1)

        # Run Object Detection on the frame
        run_infernce(frame, args, model, classes)

        # show the output frame
        cv2.imshow("SSDMobileNetv2 detections", frame)
        key = cv2.waitKey(1) & 0xFF

        # Exit if 'q' is pressed.
        if key == ord("q"):
            cap.release()
            break

# Image file.
else:
    # Read in image.
    img = cv2.imread(args.image)
    if img is not None:
        # Run Object Detection on the frame
        run_infernce(img, args, model, classes)
        # Display predictions
        cv2.imshow("SSDMobileNetv2 detections", img)
        # Quit on keypress
        key = cv2.waitKey(0)
