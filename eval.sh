#!/bin/bash
##################### Total-Text ###################################
# test_size=(640,1024)--cfglib/option
#CUDA_LAUNCH_BLOCKING=1 python eval_textBPN.py --exp_name Totaltext --checkepoch 540 --test_size 640 800 --dis_threshold 0.315 --cls_threshold 0.85 --gpu 1
###################### CTW-1500 ####################################
# test_size=(640,1024)--cfglib/option
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --exp_name Ctw1500 --checkepoch 600 --test_size 640 1024 --dis_threshold 0.375 --cls_threshold 0.8 --gpu 1

#################### MSRA-TD500 ######################################
# test_size=(640,1024)--cfglib/option
#CUDA_LAUNCH_BLOCKING=1 python eval_textBPN.py --exp_name TD500 --checkepoch 680 --dis_threshold 0.3 --cls_threshold 0.925 --gpu 1


#for ((i=55; i<=660; i=i+5))
# do CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --exp_name Ctw1500 --checkepoch $i --test_size 640 1024 --dis_threshold 0.35 --cls_threshold 0.825 --gpu 0;
#done


for ((i=55; i<=660; i=i+5))
 do CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --exp_name Totaltext --checkepoch $i --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85 --gpu 0;
 done
