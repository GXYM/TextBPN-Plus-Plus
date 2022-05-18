#!/bin/bash
cd ../

##################### eval for Ctw1500 with ResNet18 4s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet18 --scale 4 --exp_name Ctw1500 --checkepoch 105 --test_size 640 1024 --dis_threshold 0.35 --cls_threshold 0.85 --gpu 0;

##################### eval for Ctw1500 with ResNet50 1s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet50 --scale 1 --exp_name Ctw1500 --checkepoch 155 --test_size 640 1024 --dis_threshold 0.375 --cls_threshold 0.8 --gpu 0;


##################### eval for Ctw1500 with ResNet50-DCN 1s ###################################
CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net deformable_resnet50 --scale 1 --exp_name Ctw1500 --checkepoch 565 --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.925 --gpu 0;


##################### test speed for Ctw1500 ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN_speed.py --net resnet18 --scale 4 --exp_name Ctw1500 --checkepoch 570 --test_size 640 1024 --dis_threshold 0.25 --cls_threshold 0.85 --gpu 0;


##################### batch eval for Ctw1500 ###################################
#for ((i=55; i<=660; i=i+5))
# do CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet18 --scale 4 --exp_name Ctw1500 --checkepoch $i --test_size 640 1024 --dis_threshold 0.25 --cls_threshold 0.85 --gpu 1;
#done
 
