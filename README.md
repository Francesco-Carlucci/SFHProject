# SFHProject

## Introduction
The aim of this project is to build a real-time detector for the "signal for help" sign,
introduced by the Canadian Women's foundation in 2019. The presented system use google's mediapipe
library to detect hands in images and a 3D-CNN to classify the gesture performed. For this task we have
collected a dataset of signal for help videos. The classifier has been pretrained on Jester, a large public
dataset of hand gestures, and then fine-tuned through transfer learning on our specific dataset. 

## Installation

The code require pytorch and opencv, all the requirements can be found in the requirements.txty file.
It's possible to set up the environment with conda with:

     conda create --name sfh numpy opencv matplotlib pillow scipy pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
or using pip and cuda-toolkit 11.6:

    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116

## Usage

To see a demo of the system running launch demoTest.sh script, it can run either on a video or on the input stream
from the webcam:

    demoTest.sh webcam
    demoTest.sh <video_path>

//controllare script di train, inserire sezione sui dataset e link al nostro dataset

## Repository Structure
In the source folder there is all the code used for train and test of our models. The annotation folders contains
a json file with all the info needed to load the dataset and the label of each video. The datasets folder contains
both the scripts that loads the datasets and the datasets itself. In models there are the 3d implementation of the
used architectures. In results and resultSFH there are our trained models, the first one on jester, the second one
after transfer learning on our dataset. onlineTest.py is the python script that implement the whole real time system,
with mediapipe as a detector. main.py is the script we used for both training and test.

We thank Okan Köpüklü and Ahmet Gunduz for their public code, that allowed us to start this project.
