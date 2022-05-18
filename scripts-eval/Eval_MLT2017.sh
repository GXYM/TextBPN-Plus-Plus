#!/bin/bash
cd ../

##################### eval for MLT2017 with ResNet50 1s ###################################
CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet50 --scale 1 --exp_name MLT2017 --checkepoch 600 --test_size 960 2048 --dis_threshold 0.5 --cls_threshold 0.75 --gpu 1;

#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet50 --scale 1 --exp_name MLT2017 --checkepoch 600 --test_size 960 2048 --dis_threshold 0.5 --cls_threshold 0.85 --gpu 1;



##################### eval for Total-Text with ResNet50-DCN 1s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net deformable_resnet50 --scale 1 --exp_name MLT2017 --checkepoch 545 --test_size 960 2048 --dis_threshold 0.5 --cls_threshold 0.85 --gpu 0;



##################### batch eval for MLT2017 ###################################
#for ((i=100; i<=660; i=i+5))
#do 
# CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --exp_name MLT2017 --net deformable_resnet50  --scale 1 --checkepoch $i --test_size 960 2048 --dis_threshold 0.5 --cls_threshold 0.8 --gpu 0;
#done
 
 
