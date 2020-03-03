import cv2
import numpy as np
import time


def receive():

    # Load Tiny-Yolo
    net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    starting_time = time.time()
    frame_id = 0

    times = []

    pipe = 'udpsrc port=7000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink'
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER,)
    # cap = cv2.VideoCapture(
    #     'udpsrc port=7000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false',
    #     cv2.CAP_GSTREAMER,
    # )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("empty frame")
            input("wait")
            continue

        frame_time = time.time()
        # _, frame = cap.read()

        frame_id += 1

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            print(out.shape)
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

        print(
            "\nClose by pressing any button\nWindow must be focused to close, dont just click the close button!"
        )

        min_time = min(times)
        max_time = max(times)
        average_time = sum(times) / frame_id

        print(
            f"min time = {min_time} \nmax_time = {max_time} \naverage = {average_time}"
        )

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # cv2.imshow("receive", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    cv2.destroyAllWindows()


receive()

# cap = cv2.VideoCapture("videotestsrc ! videoconvert ! appsink")
# while True:
#     ret, frame = cap.read()
#     cv2.imshow("", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cv2.destroyAllWindows()
# cap.release()


# cap = cv2.VideoCapture("dev/stdin")

# input("press key to continue ")

# while True:
#     ret, frame = cap.read()
#     cv2.imshow("", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cv2.destroyAllWindows()
# cap.release()
