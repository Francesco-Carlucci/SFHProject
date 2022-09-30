#!/bin/bash

if [ $# -lt 2 ]; then
	echo 'use: demoTest <root path> <videoname> or webcam'
	sleep 5
	exit 1
fi
#C:/Users/utente/Desktop/SFHProject/minTrain
python ./source/onlineTest.py --root_path $1 --resume_path_clf resultSFH/sfh_mobilenet_2.0x_RGB_16_best.pth --video $2 --sample_duration_clf 16 --model_clf mobilenet --width_mult_clf 2 --batch_size 1 --n_classes_clf 2 --n_threads 16 --modality_clf RGB --clf_strategy raw --clf_queue_size 16 --clf_threshold_pre 1.0 --clf_threshold_final 0.15 --stride_len 1 --downsample 5 --det_counter 4 --no_cuda
