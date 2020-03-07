import cv2
import numpy as np
import time


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

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# cam = cv2.VideoCapture(0)
# time.sleep(2.0)


cvNet = cv2.dnn.readNet("weights/MobileNetSSD_V2.pb", "cfg/MobileNetSSD_V2.pbtxt")

while True:

    # _, img = cam.read()
    img = cv2.imread(
        "../test_images/visDrone/VisDrone2019-DET-val/images/0000022_01251_d_0000007.jpg"
    )
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(512, 512), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.rectangle(
                img,
                (int(left), int(top)),
                (int(right), int(bottom)),
                (23, 230, 210),
                thickness=2,
            )
            idx = int(detection[1])  # prediction class index.

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(classes[idx], score * 100)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(
                img,
                label,
                (int(left), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[idx],
                2,
            )

    # show the output frame
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
