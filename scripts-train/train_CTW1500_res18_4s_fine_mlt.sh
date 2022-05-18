#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Ctw1500 --net resnet18 --scale 4 --max_epoch 660 --batch_size 48 --gpu 0 --input_size 640 --optim Adam --lr 0.0001 --num_workers 30 --resume model/pretrain/MLT/TextBPN_resnet18_300.pth --load_memory True


#--viz --viz_freq 80
#--start_epoch 300

for ((i=55; i<=660; i=i+5))
 do CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet18 --scale 4 --exp_name Ctw1500 --checkepoch $i --test_size 640 1024 --dis_threshold 0.35 --cls_threshold 0.875 --gpu 0;
done

