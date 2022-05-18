#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name TD500 --net resnet50 --scale 1 --max_epoch 1200 --batch_size 12 --gpu 1 --input_size 640 --optim Adam --lr 0.001 --num_workers 30 --load_memory True

#--resume model/MLT2017/TextBPN_deformable_resnet50_300.pth

