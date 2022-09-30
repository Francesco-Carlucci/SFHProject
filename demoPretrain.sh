#!/bin/bash
echo $1

if [ $# -lt 1 ]; then
	echo 'use: demoPretrain <root path>'
	sleep 5
	exit 1
fi

python ./source/main.py --root_path $1 --video_path datasets/20bn-jester-v1 --annotation_path annotation_Jester/jester.json --result_path results --dataset jester --n_classes 27 --model mobilenet --width_mult 2 --train_crop random --sample_duration 16 --mean_dataset kinetics --n_epochs 40 --downsample 2 --batch_size 64 --n_threads 16 --checkpoint 1 --n_val_samples 1 --test --test_subset test

sleep 5