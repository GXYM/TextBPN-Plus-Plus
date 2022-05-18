#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Totaltext --net resnet18 --scale 4 --max_epoch 660 --batch_size 48 --gpu 0 --input_size 640 --optim Adam --lr 0.0001 --num_workers 30 --load_memory True --resume model/pretrain/MLT/TextBPN_resnet18_300.pth

#--viz --viz_freq 80
#--start_epoch 300

for ((i=55; i<=660; i=i+5))
 do CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet18 --scale 4 --exp_name Totaltext --checkepoch $i --test_size 640 1024 --dis_threshold 0.25 --cls_threshold 0.85 --gpu 0;
done

