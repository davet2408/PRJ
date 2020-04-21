import cv2
import numpy as np

# Python standard modules
import sys
import argparse
import time
import multiprocessing
import queue
from itertools import zip_longest
import csv

# Hand made modules
import yolo
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


def run_inference(model, confidence, output_layers, in_q, out_q, inference_times):
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
        if in_q.empty():
            continue
        else:
            # Start inference timing.
            inference_start_time = time.time()
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
            # Add timing info.
            inference_times.append(time.time() - inference_start_time)


def add_time_info(times):
    # Store execution time information
    with open("results/stream-time-results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("loop time", "inference time"))
        writer.writerows(times)


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
    # model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)


# Assign random colours to the classes
colours = np.random.uniform(0, 255, size=(len(classes), 3))
# Specific coloours for some classes.
colours[0] = [255.0, 0.0, 255.0]  # pink for people
colours[1] = [255.0, 3.0, 3.0]  # blue for bike
colours[2] = [3.0, 255.0, 3.0]  # green for car

frame_times = []
manger = multiprocessing.Manager()
inference_times = manger.list()

# Single element multiprocess queues for infernce.
in_q = multiprocessing.Queue(maxsize=1)
out_q = multiprocessing.Queue(maxsize=1)

detections = None

# Start the detection_process as a child process.
detection_process = multiprocessing.Process(
    target=run_inference,
    args=(model, args.conf, output_layers, in_q, out_q, inference_times),
)
# Allow main thread to exit even if detection_process is not finished.
detection_process.daemon = True
detection_process.start()

# Video capture object for retriving frames.
# cap = GStreamer_server.VideoCapture(0)
# cap = GStreamer_server.VideoCapture()

cap = cv2.VideoCapture(0)

# Create a resizable window to display detections.
cv2.namedWindow("Video Feed", cv2.WINDOW_AUTOSIZE)

# Continue reading frames until quit
while True:
    # Start loop time
    frame_start_time = time.time()

    ret, frame = cap.read()
    # No frame returned
    if not ret:
        print("empty frame")
        break

    # Discard stale frame
    try:
        in_q.get(False)
    except queue.Empty:
        pass
    # Replace frame for model to take
    in_q.put(frame)

    # Detection ready to be displayed.
    if not out_q.empty():
        detections = out_q.get()

    # Chcek for detections
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
            colours,
            frame,
            text=args.text,
        )

    # Add loop time
    frame_times.append(time.time() - frame_start_time)

    cv2.imshow("Video Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    # Exit is 'q' is pressed.
    if key == ord("q"):
        cv2.destroyAllWindows()
        break

times = zip_longest(frame_times, inference_times, fillvalue="-")
add_time_info(times)


cv2.destroyAllWindows()
cv2.waitKey(1)
if not in_q.empty():
    in_q.get(False)
# cap.stop()
cap.release()
sys.exit()
