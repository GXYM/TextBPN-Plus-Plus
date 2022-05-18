#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name ALL --net deformable_resnet50 --scale 1 --max_epoch 660 --batch_size 14 --gpu 0 --input_size 640 --optim Adam --lr 0.0001 --num_workers 30 --resume model/ArT-DCN/TextBPN_deformable_resnet50_600.pth 


#--resume model/Synthtext/TextBPN_deformable_resnet50_0.pth --viz --viz_freq 500

