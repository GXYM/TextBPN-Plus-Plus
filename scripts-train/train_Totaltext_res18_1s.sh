#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Totaltext --net resnet18 --max_epoch 660 --batch_size 24 --gpu 0 --input_size 640 --optim Adam --lr 0.001 --num_workers 30 

#--resume model/Synthtext/TextBPN_resnet50_0.pth 
#--viz --viz_freq 80
#--start_epoch 300
