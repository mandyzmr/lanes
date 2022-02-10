#!/bin/bash

# ./run.sh > log.txt 2>&1 & 
python main.py --model=deeplabv3p --backbone=resnet101 --epochs=5 \
			   --pretrained_model=global_max_mean_iou_model.pth \
			   --mode=infer --image_path='170927_064448626_Camera_6.jpg' --color_mode=True
									  