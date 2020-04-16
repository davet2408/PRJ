import cv2
import yolo
import numpy as np
import sys
import argparse
import time
from multiprocessing import Process
from multiprocessing import Queue
from video_streaming.GStreamer import GStreamer_server


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="yolov3-retrainedv2",
    help="network to use: " + str(yolo.NETWORKS.keys()),
)
parser.add_argument(
    "-c", "--conf", default=0.3, type=float, help="Set the detection threshold"
)
parser.add_argument(
    "-s", "--NMS", default=0.4, type=float, help="Non Maximum Supression threshold"
)
parser.add_argument(
    "-g", "--gpu", default=False, type=bool, help="boolean to toggle the use of gpu"
)
parser.add_argument(
    "-i",
    "--input_size",
    default=608,
    type=int,
    help="Network resolution: " + str(yolo.INPUT_SIZES),
)
parser.add_argument(
    "-t", "--text", default=False, type=bool, help="Show prediction text"
)
args = parser.parse_args()


def run_inference(model, confidence, output_layers, in_q, out_q):
    """Continually grabs frames from in_q and runs inference on that frame 
    placing the detection results in the out_q. Encapsulated in this function 
    so that it can be executed in a separate process.

    Based on code from:
    https://www.pyimagesearch.com/2017/10/16/raspberry-pi-deep-learning-object
    -detection-with-opencv/
    (accessed 17/02/20)
    """
    # Run until main process terminated.
    while True:
        # Check for new frame.
        if not in_q.empty():
            # Grab frame from queue, get dimmensions.
            frame = in_q.get()
            height, width, channels = frame.shape
            # Convert to blob.
            blob = yolo.get_blob(frame, args.input_size)
            # Pass the frame through the model.
            model.setInput(blob)
            predictions = model.forward(output_layers)
            # Format and Filter detections.
            detections = yolo.get_detections(predictions, width, height, confidence)
            # Place detections on output queue.
            out_q.put(detections)


# Load YOLO model
try:
    model, classes, output_layers = yolo.load_network(args.model, args.input_size)
except ValueError as err:
    # Invalid network parameters.
    print(str(err))
    sys.exit(1)

# Enable GPU
if args.gpu:
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# Assign random colours to the classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# Specific coloours for some classes.
colors[0] = [255.0, 0.0, 255.0]  # pink for people
colors[1] = [255.0, 3.0, 3.0]  # blue for bike
colors[2] = [3.0, 255.0, 3.0]  # green for car

# Timing utilities for inference
times = []

# Single element multiprocess queues for infernce.
in_q = Queue(maxsize=1)
out_q = Queue(maxsize=1)

detections = None

# Start the inference (child) process.
p = Process(target=run_inference, args=(model, args.conf, output_layers, in_q, out_q,))
# Allow main thread to exit even if process is not finished.
p.daemon = True
p.start()

# cap = cv2.VideoCapture(0)
cap = GStreamer_server.VideoCapture()

cv2.namedWindow("Video Feed", cv2.WINDOW_AUTOSIZE)

while True:

    ret, frame = cap.read()
    if not ret:
        print("empty frame")
        continue

    # Model waiting for next frame.
    if in_q.empty():
        in_q.put(frame)
    # Detection ready to be displayed.
    if not out_q.empty():
        detections = out_q.get()

    if detections is not None:
        start_time = time.time()

        class_ids, confidences, boxes = detections
        # Non-maximal supresion to remove duplicate detections
        nms_indexes = cv2.dnn.NMSBoxes(boxes, confidences, args.conf, args.NMS)
        # Place detections on frame
        yolo.draw_bounding_boxes(
            class_ids,
            confidences,
            boxes,
            nms_indexes,
            classes,
            colors,
            frame,
            text=args.text,
        )

    cv2.imshow("Video Feed", frame)

    key = cv2.waitKey(10) & 0xFF
    # Exit is 'q' is pressed.
    if key == ord("q"):
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
