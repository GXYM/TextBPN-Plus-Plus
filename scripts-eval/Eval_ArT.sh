#!/bin/bash
cd ../
##################### eval for ArT with ResNet50 1s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet50 --scale 1 --exp_name ArT --checkepoch 605 --test_size 960 2880 --dis_threshold 0.4 --cls_threshold 0.4 --gpu 0;


##################### eval for ArT with ResNet50-DCN 1s ###################################
CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net deformable_resnet50 --scale 1 --exp_name ArT --checkepoch 480 --test_size 960 2880 --dis_threshold 0.4 --cls_threshold 0.8 --gpu 0;


##################### batch eval for ArT ###################################
#for ((i=660; i>=300; i=i-5));
#do 
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --exp_name ArT --net deformable_resnet50 --checkepoch $i --test_size 960 2880 --dis_threshold 0.45 --cls_threshold 0.8 --gpu 0;
#done
 
