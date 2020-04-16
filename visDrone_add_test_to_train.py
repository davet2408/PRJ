import os
import sys
import cv2
import numpy as np
import random


def add_line(path, line):
    # write annotation information to new file
    with open(path, "a") as file:
        file.write(line + "\n")


# File paths
train_path = "data/obj/"
image_dir = "test_images/visDrone/VisDrone2019-DET-test-dev/images/"
obj = "test_images/visDrone/yolo-training-server/training-files/obj/"
train = "test_images/visDrone/yolo-training-server/training-files/train.txt"
test = "test_images/visDrone/yolo-training-server/VisDrone-test-dev-yolo/"
test_voc = "test_images/visDrone/yolo-training-server/VisDrone-test-dev-voc/"

# New directory for smaller test set
test_images500 = "test_images/visDrone/yolo-training-server/test_images500/"
os.mkdir(test_images500)

# Original test set
og_test = os.listdir(test)
# Random selection
random.shuffle(og_test)

# Leave only 500 test images
selection = og_test[500:]
new_test_set = og_test[:500]

print(len(os.listdir(test)))

# Loop through random selection and move to training set
for annotation in selection:
    # Copy image to obj file
    image_file = annotation.split(".txt")[0] + ".jpg"
    image_path = os.path.join(image_dir, image_file)
    img = cv2.imread(image_path)
    cv2.imwrite(obj + image_file, img)
    # Move annotations from test to train
    add_line(train, train_path + image_file)
    os.rename(test + annotation, obj + annotation)
    os.remove(test_voc + annotation)

print(len(os.listdir(test)))

# Add new test set images directory
for annotation in new_test_set:
    image_file = annotation.split(".txt")[0] + ".jpg"
    image_path = os.path.join(image_dir, image_file)
    img = cv2.imread(image_path)
    cv2.imwrite(test_images500 + image_file, img)
