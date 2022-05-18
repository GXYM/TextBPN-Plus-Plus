#!/bin/bash
cd ../

##################### eval for TD500 with ResNet18 4s ###################################
CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet18 --scale 4 --exp_name TD500 --checkepoch 1135 --test_size 640 960 --dis_threshold 0.35 --cls_threshold 0.9 --gpu 0; #--viz

##################### eval for TD500 with ResNet50 1s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet50 --scale 1 --exp_name TD500 --checkepoch 1070 --test_size 640 960 --dis_threshold 0.35 --cls_threshold 0.875 --gpu 0;


##################### eval for TD500 with ResNet50-DCN 1s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net deformable_resnet50 --scale 1 --exp_name TD500 --checkepoch 355 --test_size 640 960 --dis_threshold 0.3 --cls_threshold 0.95 --gpu 0;


##################### test speed for TD500 ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN_speed.py --net resnet18 --scale 4 --exp_name TD500 --checkepoch 1135 --test_size 640 960 --dis_threshold 0.35 --cls_threshold 0.9 --gpu 1;


##################### batch eval for TD500 ###################################
#for ((i=55; i<=1200; i=i+5))
# do CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet18 --scale 4 --exp_name TD500 --checkepoch $i --test_size 640 960 --dis_threshold 0.35 --cls_threshold 0.9 --gpu 1;
#done
 
