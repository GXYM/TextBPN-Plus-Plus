#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Synthtext --net deformable_resnet50 --scale 1 --max_epoch 1 --batch_size 14 --lr 0.0001 --gpu 1 --input_size 640 --save_freq 1 --num_workers 30 --viz --viz_freq 1000

