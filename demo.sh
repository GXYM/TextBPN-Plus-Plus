#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 python3 demo.py --net resnet18 --scale 4 --exp_name TD500 --checkepoch 1135 --test_size 640 960 --dis_threshold 0.35 --cls_threshold 0.9 --gpu 0 --img_root ./Test1  --viz
