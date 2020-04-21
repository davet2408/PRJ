import cv2
import numpy as np
import argparse
import yolo
import sys

#  Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="yolov3-retrainedv2",
    help="network to use: " + str(yolo.NETWORKS.keys()),
)
parser.add_argument(
    "--video", default=None, help="Relative path to video file to run yolo on."
)
parser.add_argument(
    "--image",
    default="../test_images/hard_test_images/9999947_00000_d_0000026.jpg",
    help="Relative path to image file to run yolo on.",
)
parser.add_argument(
    "-c", "--conf", default=0.2, type=float, help="Set the detection threshold"
)
parser.add_argument(
    "-s", "--NMS", default=0.6, type=float, help="Non Maximum Supression threshold"
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


def image_detection(args, model, classes, output_layers):

    img = cv2.imread(args.image)
    if img is not None:
        height, width, channels = img.shape

        blob = yolo.get_blob(img, args.input_size)

        model.setInput(blob)
        layer_outputs = model.forward(output_layers)
        # Network prediction information
        class_ids, confidences, boxes = yolo.get_detections(
            layer_outputs, width, height, args.conf
        )
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
            img,
            text=args.text,
        )

        # Display image with predictions.
        cv2.imshow(f"{args.model} {args.input_size}x{args.input_size} detections", img)
        cv2.waitKey(0)

    else:
        print(f"No image found at - {args.image}")


def video_detection(args, model, classes, output_layers):

    if args.video == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"No video found at {args.video}")
            sys.exit(1)

        height, width, channels = frame.shape
        # Convert to blob.
        blob = yolo.get_blob(frame, args.input_size)
        # Pass the frame through the model.
        model.setInput(blob)
        predictions = model.forward(output_layers)
        # Format and Filter detections.
        class_ids, confidences, boxes = yolo.get_detections(
            predictions, width, height, args.conf
        )
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

        cv2.imshow("Video Feed", frame)

        key = cv2.waitKey(10) & 0xFF
        # Exit is 'q' is pressed.
        if key == ord("q"):
            cap.release()
            break


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
colours = np.random.uniform(0, 255, size=(len(classes), 3))
# Specific coloours for some classes.
colours[0] = [255.0, 0.0, 255.0]  # pink for people
colours[1] = [255.0, 3.0, 3.0]  # blue for bike
colours[2] = [3.0, 255.0, 3.0]  # green for car
# colours[8] = [3.0, 255.0, 255.0]  # yellow for motor bike


if args.video is None:
    img = image_detection(args, model, classes, output_layers)
    # save image
    # cv2.imwrite("detection_result.jpg", img)

else:
    video_detection(args, model, classes, output_layers)


cv2.destroyAllWindows()
