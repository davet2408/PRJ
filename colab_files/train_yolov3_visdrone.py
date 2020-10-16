# -*- coding: utf-8 -*-
"""Train-yolov3-visDrone.ipynb

Based on method from: https://github.com/theAIGuysCode/YOLOv3-Cloud-Tutorial
"""

# clone AlexeyAB darknet repo
!git clone https://github.com/AlexeyAB/darknet

# check for CUDA
!/usr/local/cuda/bin/nvcc --version

# Commented out IPython magic to ensure Python compatibility.
# change makefile to have GPU and OPENCV enabled
# %cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile

# build darknet
!make

# get pretrained weights
!wget https://pjreddie.com/media/files/yolov3.weights

# second prediction
!./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg

# Commented out IPython magic to ensure Python compatibility.
# %cd ..
from google.colab import drive
drive.mount('/content/gdrive')

# create smybolic link to google drive 
!ln -s /content/gdrive/My\ Drive/ /drive
!ls /drive

# google drive yolov3 folder
!ls /drive/yolov3-training

!ls

# Commented out IPython magic to ensure Python compatibility.
# %cd darknet/
# copy over obj.zip
!cp /drive/yolov3-training/obj.zip ../

# unzip into /darknet/data/
!unzip ../obj.zip -d data/

# custom cfg file
#!cp cfg/yolov3.cfg /drive/yolov3-training/yolov3_custom.cfg

# local download
#download('cfg/yolov3.cfg')

# copy back the cfg file
!cp /drive/yolov3-training/yolov3-obj.cfg ./cfg

# get cfg from local
#%cd cfg
#upload()
#%cd ..

# copy names and data over
!cp /drive/yolov3-training/obj.names ./data
!cp /drive/yolov3-training/obj.data  ./data

# upload names and data
#%cd data
#upload()
#%cd ..

!cp /drive/yolov3-training/train.txt ./data
!cp /drive/yolov3-training/val.txt  ./data

# check everythings in the data folder 
!ls data/

# upload pretrained convolutional layer weights
!wget http://pjreddie.com/media/files/darknet53.conv.74

# train from scratch
!./darknet detector train data/obj.data cfg/yolov3-obj.cfg darknet53.conv.74 -dont_show -map

# continue training from last added weights
!./darknet detector train data/obj.data cfg/yolov3-obj.cfg /drive/yolov3-training/backup/yolov3-obj_last.weights -dont_show -map