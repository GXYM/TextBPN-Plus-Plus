#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Ctw1500 --net resnet50 --scale 1 --max_epoch 660 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.0001 --num_workers 30 --resume model/pretrain/MLT/TextBPN_resnet50_300.pth --load_memory True 


#--viz --viz_freq 80
#--start_epoch 300

#for ((i=55; i<=660; i=i+5))
# do CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet50 --scale 1 --exp_name Ctw1500 --checkepoch $i --test_size 640 1024 --dis_threshold 0.375 --cls_threshold 0.8 --gpu 0;
#done

