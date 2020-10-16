# -*- coding: utf-8 -*-
"""coco_data.ipynb

Based on method from: https://github.com/ivangrov/Datasets-ASAP/tree/master/%5BPart%203%5D%20COCO%20Dataset
"""

!ls

# download the validation dataset from coco
!wget http://images.cocodataset.org/zips/val2017.zip

# download the annotations that go with the data
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# further annotation information
!wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !unzip stuff_annotations_trainval2017.zip

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !unzip annotations_trainval2017.zip

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !unzip val2017.zip

!pip install gluoncv
!pip install mxnet

from gluoncv import data, utils
from matplotlib import pyplot as plt

val_dataset = data.COCODetection('.',splits=['instances_val2017'])
print('Num of validation images:', len(val_dataset))

val_image, val_label = val_dataset[0]
bounding_boxes = val_label[:, :4]
class_ids = val_label[:, 4:5]
print(val_label.shape)
print('Image size (height, width):', val_image.shape)
print('Number of objects:', bounding_boxes.shape[0])

utils.viz.plot_bbox(val_image.asnumpy(), bounding_boxes, scores=None,
                    labels=class_ids, class_names=val_dataset.classes)
plt.show()

# all 80 classes in order
coco_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
print(len(coco_names))

# Test that the images bouding box information has been translated correctly
val_image, val_label = val_dataset[0]
bounding_boxes = val_label[:, :4]
class_ids = val_label[:, 4:5]

for i in range(len(bounding_boxes)):
  class_name = coco_names[int(class_ids[i][0])]
  x1 = int(bounding_boxes[i][0])
  y1 = int(bounding_boxes[i][1])
  x2 = int(bounding_boxes[i][2])
  y2 = int(bounding_boxes[i][3])

  line = f"{class_name} {str(x1)} {str(y1)} {str(x2)} {str(y2)}"
  print(line)

# remove directories if they already exisit
!rm -r ground-truth/
!rm -r images/

# create new directories
!mkdir ground-truth
!mkdir images
!ls

import os
import cv2

def add_bb_info(path, line):
  with open(path, 'a') as file:
    file.write(line + "\n")


# folders to store image information
ground_truth_folder = 'ground-truth'
images_folder = 'images'

n = 0

num_of_images = 500

for i in range(num_of_images):

  name = str(n)
  ground_truth_path = os.path.join(ground_truth_folder, name)+'.txt'
  images_path = os.path.join(images_folder, name) +'.jpg'

  val_image, val_label = val_dataset[i]
  bounding_boxes = val_label[:, :4]
  class_ids = val_label[:, 4:5]

  image = val_image.asnumpy()
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  for j in range(len(bounding_boxes)):
    class_name = coco_names[int(class_ids[j][0])]
    x1 = int(bounding_boxes[j][0])
    y1 = int(bounding_boxes[j][1])
    x2 = int(bounding_boxes[j][2])
    y2 = int(bounding_boxes[j][3])

    line = f"{class_name} {str(x1)} {str(y1)} {str(x2)} {str(y2)}"

    add_bb_info(ground_truth_path, line)
    
    cv2.imwrite(images_path,image)


  n += 1

val_im, _ = val_dataset[1]
print(type(val_im))

!zip -r coco_test.zip ground-truth images

from google.colab import drive
drive.mount('/content/gdrive')

