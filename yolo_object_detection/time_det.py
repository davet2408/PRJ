"""
This program provides Object Detection to a video feed utilising threads to 
try and enhance performance. 

By default the program will execute with the optimal settings for Object 
Detection from a GStreamer stream assumed to be from a drones perspective. 
The command line arguments can be used to adjust performance and even switch 
to a webcame view for testing.


Author: David Temple
Date: 02/03/2020
"""
# OpenCV module installed from https://github.com/opencv/opencv
import cv2

# numpy module installed via pip https://numpy.org
import numpy as np

# Python standard library modules
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


#### Command line arguments ####
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="yolov3-retrainedv2",
    help="network to use: " + str(yolo.NETWORKS.keys()),
)
parser.add_argument(
    "-c", "--conf", default=0.1, type=float, help="Set the detection threshold"
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
    "-t", "--text", default=False, type=bool, help="boolean to toggle prediction text"
)
parser.add_argument(
    "-w",
    "--webcam",
    default=False,
    type=bool,
    help="boolean to toggle webcam as input stream",
)
args = parser.parse_args()


def initialise_detector(model, size, gpu):
    """Attempts to set up the Object Detection model, will exit and display 
    issue if invalid parameters are given."""
    try:
        model, _, output_layers = yolo.load_network(model, size)
    except ValueError as err:
        # Invalid network parameters.
        print(str(err))
        sys.exit(1)

    # # Enable GPU
    if gpu:
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    return model, output_layers


def run_inference(model, confidence, in_q, out_q, size, inference_times):
    """Continually grabs frames from in_q and runs inference on that frame 
    placing the detection results in the out_q. Encapsulated in this function 
    so that it can be executed in a separate process.

    Based on code from:
    https://www.pyimagesearch.com/2017/10/16/raspberry-pi-deep-learning-object
    -detection-with-opencv/
    (accessed 17/02/20)
    """

    model, output_layers = initialise_detector(args.model, args.input_size, args.gpu)

    # Run until main process terminated.
    while True:
        # Check for new frame.
        if not in_q.empty():
            # Start inference timing.
            inference_start_time = time.time()
            # Grab frame from queue, get dimmensions.
            count, frame = in_q.get()
            height, width, channels = frame.shape
            # Convert to blob.
            blob = yolo.get_blob(frame, size)

            # Pass the frame through the model.
            model.setInput(blob)
            predictions = model.forward(output_layers)
            inference_time = time.time() - inference_start_time

            # Format and Filter detections.
            detections = yolo.get_detections(predictions, width, height, confidence)
            # Place detections on output queue.
            out_q.put(detections)

            post_processing_time = time.time() - inference_start_time - inference_time
            # Add timing info.
            inference_times.append([inference_time, post_processing_time, count])


def add_time_info(times):
    # Store execution time information
    with open("results/stream-time-results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ("frame", "loop time", "inference time", "post_processing_time", "frame")
        )
        writer.writerows(times)


def generate_colours(size):
    """Generate a set of random colours and assign some shared class colours."""
    colours = np.random.uniform(0, 255, size=(size, 3))
    # Specific coloours for some classes shared between all models.
    colours[0] = [255.0, 0.0, 255.0]  # pink for people
    colours[1] = [255.0, 3.0, 3.0]  # blue for bike
    colours[2] = [3.0, 255.0, 3.0]  # green for car

    return colours


def display_loop(cap, detections, in_q, out_q, args):
    """Continually read frames and add them to the detector queue. Display 
    predictions when available. Wait for exit command.
    """

    # Frame counter
    count = 0
    # Continue reading frames until quit
    while True:

        frame_start_time = time.time()

        ret, frame = cap.read()
        # No frame returned
        if not ret:
            print("empty frame")
            break

        count += 1

        # Discard stale frame
        try:
            in_q.get(False)
        except queue.Empty:
            pass
        # Replace frame for model to take
        in_q.put((count, frame))

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

        # Show the frame with detections.
        cv2.imshow("Video Feed", frame)

        # Add loop time
        loop_time = time.time() - frame_start_time
        frame_times.append([count, loop_time])

        # Set 100 fps max.
        key = cv2.waitKey(10) & 0xFF
        # Exit if 'q' is pressed.
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    # Change process from forked to spawned.
    multiprocessing.set_start_method("spawn")

    # Video capture object for retriving frames.
    if args.webcam:
        cap = GStreamer_server.VideoCapture(0)
    else:
        cap = GStreamer_server.VideoCapture()
        # cap = cv2.VideoCapture(0)  # can be used for video files.

    # Generate class list for model used.
    classes = yolo.get_classes(args.model)
    # Assign random colours to the classes
    colours = generate_colours(len(classes))

    # Single element multiprocess queues for infernce.
    in_q = multiprocessing.Queue(maxsize=1)
    out_q = multiprocessing.Queue(maxsize=1)

    detections = None

    # Timing utilities.
    frame_times = []
    # Process sharable list.
    manger = multiprocessing.Manager()
    inference_times = manger.list()

    # Start the detection_process as a child process.
    detection_process = multiprocessing.Process(
        target=run_inference,
        args=(args.model, args.conf, in_q, out_q, args.input_size, inference_times),
    )
    # Allow main thread to exit even if detection_process is not finished.
    detection_process.daemon = True
    detection_process.start()

    # Create a resizable window to display detections.
    cv2.namedWindow("Video Feed", cv2.WINDOW_AUTOSIZE)

    # Loop main thread until canceled
    display_loop(cap, detections, in_q, out_q, args)

    # Write execution time information to CSV file
    l1 = [x[0] for x in frame_times]
    l2 = [x[1] for x in frame_times]
    l3 = [x[0] for x in inference_times]
    l4 = [x[1] for x in inference_times]
    l5 = [x[2] for x in inference_times]
    times = zip_longest(l1, l2, l3, l4, l5, fillvalue="-")
    add_time_info(times)

    # Clean up resources
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if not in_q.empty():
        in_q.get(False)
    cap.stop()
    # cap.release() # if using video file
    sys.exit()
