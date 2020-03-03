import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
# Load Tiny-Yolo
# net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Stream webcam
# cap = cv2.VideoCapture(0)

# Video stream
cap = cv2.VideoCapture("test_videos/birds_eye.mov")

starting_time = time.time()
frame_id = 0

times = []

while True:
    frame_time = time.time()
    _, frame = cap.read()

    # video stream
    # frame = cv2.resize(frame, None, fx=0.2, fy=0.2)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (608, 608), (0, 0, 0), True, crop=False
    )

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
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

    elapsed_time = time.time() - starting_time
    infer_time = time.time() - frame_time
    times.append(infer_time)

    # print(infer_time)

    # fps = frame_id / elapsed_time
    # print("\nFPS: ", fps)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


print(
    "\nClose by pressing any button\nWindow must be focused to close, dont just click the close button!"
)

min_time = min(times)
max_time = max(times)
average_time = sum(times) / frame_id

print(f"min time = {min_time} \nmax_time = {max_time} \naverage = {average_time}")

cap.release()
cv2.destroyAllWindows()
