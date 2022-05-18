from easydict import EasyDict
import torch
import os

config = EasyDict()


# Normalize image
config.means = (0.485, 0.456, 0.406)
config.stds = (0.229, 0.224, 0.225)

config.gpu = "1"

# Experiment name #
config.exp_name = "Synthtext"

# dataloader jobs number
config.num_workers = 24

# batch_size
config.batch_size = 12

# training epoch number
config.max_epoch = 200

config.start_epoch = 0

# learning rate
config.lr = 1e-4

# using GPU
config.cuda = True

config.output_dir = 'output'

config.input_size = 640

# max polygon per image
# synText, total-text:64; CTW1500: 64; icdar: 64;  MLT: 32; TD500: 64.
config.max_annotation = 64

# adj num for graph
config.adj_num = 4

# control points number
config.num_points = 20

# use hard examples (annotated as '#')
config.use_hard = True

# Load data into memory at one time
config.load_memory = False

# prediction on 1/scale feature map
config.scale = 1

# # clip gradient of loss
config.grad_clip = 25

# demo tcl threshold
config.dis_threshold = 0.3

config.cls_threshold = 0.8

# Contour approximation factor
config.approx_factor = 0.004


def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    # print(config.gpu)
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
