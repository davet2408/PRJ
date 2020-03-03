from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
import cv2


def classify_frame(net, output_layers, inputQueue, outputQueue):
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            frame = inputQueue.get()
            # frame = cv2.resize(frame, (416, 416))
            blob = cv2.dnn.blobFromImage(
                frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
            )
            # set the blob as input to our deep learning object
            # detector and obtain the detections
            net.setInput(blob)
            detections = net.forward(output_layers)
            # write the detections to the output queue
            outputQueue.put(detections)


# Load Tiny-Yolo
net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")

# Load class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# Assign random colours to the classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Timing utilities for inference
starting_time = time.time()
frame_id = 0

times = []

# Input queue to collect frames from the stream
inputQueue = Queue(maxsize=1)
# Output queue to collect detection results
outputQueue = Queue(maxsize=1)
detections = None


# Construct child process
p = Process(target=classify_frame, args=(net, output_layers, inputQueue, outputQueue,))
p.daemon = True
p.start()

# GStreamer pipeline for udp stream
pipe = 'udpsrc port=7000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false'
cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER,)


while True:

    cap.grab()
    cap.grab()
    cap.grab()

    ret, frame = cap.read()
    if not ret:
        print("empty frame")
        input("wait")
        continue

    frame_time = time.time()
    frame_id += 1

    height, width, channels = frame.shape

    # If net is waiting for input
    if inputQueue.empty():
        inputQueue.put(frame)
    # If there is an outstanding detection to be shown
    if not outputQueue.empty():
        detections = outputQueue.get()

    if detections is not None:
        class_ids = []
        confidences = []
        boxes = []
        for out in detections:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
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

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    label + " " + str(round(confidence, 2)),
                    (x, y + 30),
                    font,
                    1,
                    color,
                    3,
                )

        # elapsed_time = time.time() - starting_time
        # infer_time = time.time() - frame_time
        # times.append(infer_time)

        cv2.imshow("Image", frame)

    # min_time = min(times)
    # max_time = max(times)
    # average_time = sum(times) / frame_id

    # print(
    #     f"min time = {min_time} \nmax_time = {max_time} \naverage = {average_time}"
    # )

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
