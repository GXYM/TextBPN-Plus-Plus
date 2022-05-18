#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Synthtext --net resnet50 --scale 1 --max_epoch 1 --batch_size 12 --lr 0.001 --gpu 0 --input_size 640 --save_freq 1 --num_workers 30 

#--resume model/Synthtext/TextBPN_resnet50_0.pth --start_epoch 1 --viz --viz_freq 10000
