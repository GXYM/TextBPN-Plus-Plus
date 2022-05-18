#!/bin/bash
cd ../

##################### eval for Total-Text with ResNet18 4s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet18 --scale 4 --exp_name Totaltext --checkepoch 570 --test_size 640 1024 --dis_threshold 0.25 --cls_threshold 0.85 --gpu 0;

##################### eval for Total-Text with ResNet50 1s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet50 --scale 1 --exp_name Totaltext --checkepoch 285 --test_size 640 1024 --dis_threshold 0.325 --cls_threshold 0.85 --gpu 0;


##################### eval for Total-Text with ResNet50-DCN 1s ###################################
CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net deformable_resnet50 --scale 1 --exp_name Totaltext --checkepoch 480 --test_size 640 1024 --dis_threshold 0.325 --cls_threshold 0.9 --gpu 0;


##################### test speed for Total-Text ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN_speed.py --net resnet18 --scale 4 --exp_name Totaltext --checkepoch 570 --test_size 640 1024 --dis_threshold 0.25 --cls_threshold 0.85 --gpu 0;


##################### batch eval for Total-Text ###################################
#for ((i=55; i<=660; i=i+5))
# do CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet18 --scale 4 --exp_name Totaltext --checkepoch $i --test_size 640 1024 --dis_threshold 0.25 --cls_threshold 0.85 --gpu 1;
#done
 
