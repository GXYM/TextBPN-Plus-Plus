#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Ctw1500 --net resnet50 --scale 1 --max_epoch 660 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.001 --num_workers 30 --viz --viz_freq 80

#--resume model/Ctw1500/TextBPN_resnet50_390.pth --start_epoch 665
#--resume model/Synthtext/TextBPN_resnet50_0.pth 
#--viz --viz_freq 80
#--start_epoch 300

###### test eval ############
for ((i=255; i<=660; i=i+5))
 do CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --exp_name Ctw1500 --checkepoch $i --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.825 --gpu 0;
done



