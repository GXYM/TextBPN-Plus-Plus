#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Totaltext --net resnet50 --scale 1 --max_epoch 660 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.001 --num_workers 30
#--resume model/MLT2017/TextBPN_resnet50_100.pth 
#--start_epoch 300
#--viz 

###### test eval ############
for ((i=100; i<=660; i=i+5))
 do CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --exp_name Totaltext --checkepoch $i --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85 --gpu 0;
 done
