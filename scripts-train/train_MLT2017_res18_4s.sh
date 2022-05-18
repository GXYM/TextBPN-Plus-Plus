#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name MLT2017 --net resnet18 --scale 4 --max_epoch 660 --batch_size 48 --gpu 0 --input_size 640 --optim Adam --lr 0.001 --num_workers 30

#--start_epoch 276


# --resume model/MLT2017/TextBPN_resnet50_200.pth --viz --viz_freq 500
