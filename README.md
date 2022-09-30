# SFHProject

## source: 
script for train and test, and all the functions they use

## demoPretrain: 
shell script to pretrain mobilenet on jester from scratch

## demoTest: 
shell script to test real time our mobilenet model. Use: demoTest-sh  <root path> then:

    - webcam : in order to test with pc webcam
    - video path : path of the video to test

## requirements:
list of packets needed with their version. It's possible to set up the environments
also through:

    conda --name <name> pytorch torchvision cudatoolkit=11.6 numpy opencv matplotlib pillow scipy -c pytorch -c conda-forge