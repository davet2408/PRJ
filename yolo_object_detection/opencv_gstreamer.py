import numpy as np
import cv2


def receive():

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
            continue

        cv2.imshow("receive", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


receive()

