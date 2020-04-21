"""
This module provides an abstractions of the main functions required to run YOLO 
Object detection models with OpenCV. This has been developed to help reduce 
repitition of code and can be imported by other modules in this directory.

There is only a limited number of ways to interpret the yolo output from OpenCV
so therefore many sources have been considered and used in the development of 
this module.

Sections of this code is based on implimentations from these sources:
https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/ 
(accessed 12/01/20)

https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3
-with-opencv-python-c/     
(accessed 20/01/20)

https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/     
(accessed 12/01/20)

Author: David Temple
Date: 02/03/2020
"""

import cv2
import numpy as np

# List of valid networks and the detection classes.
NETWORKS = {
    "yolov3": "coco",
    "yolov3-tiny": "coco",
    "yolov3-retrainedv1": "obj_retrainedv1",
    "yolov3-retrainedv2": "obj_retrainedv2",
}
# Valid input sizes to the networks.
INPUT_SIZES = [320, 416, 608]
# Yolo scale factor
SCALE_FACTOR = 1 / 255


def _is_valid_network(network, size):
    """Check that the network paramters are valid"""
    if network not in NETWORKS or size not in INPUT_SIZES:
        return False
    else:
        return True


def load_network(network, size):
    """Initalise the requested Object Detection model.
    
    Arguments:
        network {[str]} -- The model to be loaded.
        size {[int]} -- Resolution of the model to be loaded.
    
    Raises:
        ValueError: If model name or resolution are not found.
    
    Returns:
        [dnn, list[str], List[int]] -- Object detection model, list of classes
        the model can detect, list of output layers from the network.
    """
    if _is_valid_network(network, size):
        # Initialize the network from weights and cfg files.
        net = cv2.dnn.readNet(f"weights/{network}.weights", f"cfg/{network}.cfg")

        # Load class labels for relavant file.
        classes = []
        with open(f"{NETWORKS[network]}.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        # Get the output layers from each scale (unconnected layers).
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return (net, classes, output_layers)

    else:
        # Incorrect network parameters
        raise ValueError(
            f"Incorrect netowk type or size : {network} - {size}\n"
            f"networks: {[net for net in NETWORKS.keys()]}\n"
            f"sizes: {str(INPUT_SIZES)}"
        )


def get_blob(frame, size):
    """Wrapper around OpenCV blobFromImage function for ease of use.
    
    Arguments:
        frame {[ndarry]} -- Frame to be converted to blob.
        size {[int]} -- Dimensions to resize to.
    
    Returns:
        [ndarry] -- Blob ready to input to model.
    """
    return cv2.dnn.blobFromImage(
        frame, SCALE_FACTOR, (size, size), (0, 0, 0), swapRB=True, crop=False,
    )


def get_detections(layer_outputs, frame_width, frame_height, confidence_threshold):
    """Interpret the output of the object detector, filtering out week 
    detections. The remaining are converted back to co-ordinate results 
    from their regularised form and stored ready for NMS.

    Based on code from:
    https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
    (accessed 12/01/20)
    and 
    https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3
    -with-opencv-python-c/     
    (accessed 20/01/20)
    
    Arguments:
        layer_outputs {[ndarry]} -- Raw predictions to be processed.
        frame_width {[int]} -- Origonal width of frame passed to model.
        frame_height {[int]} -- Origonal height of frame passed to model.
        confidence_threshold {[float]} -- Minimum value for positive prediction.
    
    Returns:
        [(list[int], list[float], list[int,int,int,int])] -- ID of classes detected,
        confidence scores of those detections, bounding box co-ordinates.
    """
    class_ids = []
    bounding_boxes = []
    confidence_scores = []
    for outputs in layer_outputs:
        for detection in outputs:
            prediction_scores = detection[5:]
            class_id = np.argmax(prediction_scores)
            confidence = prediction_scores[class_id]
            if confidence > confidence_threshold:
                # Convert to frame co-ordinates
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                # Top left corner of bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                bounding_boxes.append([x, y, w, h])
                confidence_scores.append(float(confidence))

    return (class_ids, confidence_scores, bounding_boxes)


def draw_bounding_boxes(
    class_ids,
    confidence_scores,
    bounding_boxes,
    indexes,
    classes,
    colours,
    frame,
    text=True,
):
    """Filter results of NMS and then draw bounding boxes onto the origonal
    frame/image. Optionally add class name and detection confidence to bouding
    boxs.


    Based on code from:
        https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
        (accessed 12/01/20)

    Arguments:
        class_ids {list[int]} -- List of class ids that have been detected.
        confidence_scores {list[float]} -- Confidences of detections.
        bounding_boxes {list[int,int,int,int]} -- List of bounding boxes to be filtered.
        indexes {[int]} -- Indexes of boxes that are left after NMS.
        classes {[str]} -- list of classes that can be detected.
        colours {[(int,int,int)]} -- List of colours to use for different classes.
        frame {[ndarry]} -- The frame to add predictions to.

    Keyword Arguments:
        text {bool} -- Toggle text on detection results. (default: {True})
    """
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 1
    line_thickness = 2

    for box_idx in range(len(bounding_boxes)):
        # Only use boxes that passed NMS so in indexes.
        if box_idx in indexes:
            # Draw bounding box
            x, y, w, h = bounding_boxes[box_idx]
            label = classes[class_ids[box_idx]]
            colour = colours[class_ids[box_idx]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, line_thickness)
            # Add text
            if text:
                confidence = confidence_scores[box_idx]
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x, y - 3),
                    font,
                    font_size,
                    colour,
                    line_thickness,
                )
