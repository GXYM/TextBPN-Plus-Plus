import os
import gc
import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset import AllTextJson
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import Augmentation
from cfglib.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from cfglib.option import BaseOptions
from util.visualize import visualize_network_output
from util.summary import LogSummary
from util.shedule import FixLR
from torch.amp import GradScaler, autocast


lr = None
train_step = 0

class WarmupScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, base_lr, final_lr, after_scheduler=None):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.after_scheduler = after_scheduler
        super(WarmupScheduler, self).__init__(optimizer)

    def get_last_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_lr = self.base_lr + (self.final_lr - self.base_lr) * (self.last_epoch / self.warmup_epochs)
            return [warmup_lr]
        if self.after_scheduler:
            if self.last_epoch == self.warmup_epochs:
                self.after_scheduler.base_lrs = self.final_lr
            return self.after_scheduler.get_last_lr()
        return [self.final_lr]

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            warmup_lr = self.base_lr + (self.final_lr - self.base_lr) * (self.last_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            if self.after_scheduler:
                if self.last_epoch == self.warmup_epochs:
                    self.after_scheduler.base_lrs = self.final_lr
                self.after_scheduler.step()
        self.last_epoch += 1



def save_model(model, epoch, lr, optimizer):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'TextBPN_{}_{}.pth'.format(model.module.backbone_name, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.mgpu else model.module.state_dict(),
        # 'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)

def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    # state_dict = torch.load(model_path)
    state_dict = torch.load(model_path, map_location=cfg.device, weights_only=True) 
    model.load_state_dict(state_dict['model'])

def _parse_data(inputs):
    input_dict = {}
    inputs = list(map(lambda x: to_device(x), inputs))
    input_dict['img'] = inputs[0]
    input_dict['train_mask'] = inputs[1]
    input_dict['tr_mask'] = inputs[2]
    input_dict['distance_field'] = inputs[3]
    input_dict['direction_field'] = inputs[4]
    input_dict['weight_matrix'] = inputs[5]
    input_dict['gt_points'] = inputs[6]
    input_dict['proposal_points'] = inputs[7]
    input_dict['ignore_tags'] = inputs[8]

    return input_dict

def train(model, train_loader, criterion, scheduler, optimizer, epoch, scaler, use_amp=True, accum_grad_iters=1):
    global train_step

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()

    print('Epoch: {} : LR = {}'.format(epoch, scheduler.get_last_lr()))

    for i, inputs in enumerate(train_loader):
        data_time.update(time.time() - end)
        train_step += 1
        input_dict = _parse_data(inputs)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            # with torch.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            output_dict = model(input_dict)
            loss_dict = criterion(input_dict, output_dict, eps=epoch+1)
            loss = loss_dict["total_loss"]
            loss /= accum_grad_iters #TODO: not affect loss_dict values for logging

            if torch.isnan(loss).any() and loss.item()>16.0:
                print(f"loss: {loss}")
                continue

            try:
                # after_train_step()
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
                # Clip the gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                # update gradients every accum_grad_iters iterations
                if (i + 1) % accum_grad_iters == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()                     
                    else:    
                        optimizer.step()
                    optimizer.zero_grad()
            except:
                print("loss gg")
                for (k, v) in loss_dict.items():
                    print_inform += " {}: {:.4f} ".format(k, v.item())
                continue
        
        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        scheduler.step()
        
        # 只在主进程上打印信息
        if cfg.viz and (i % cfg.viz_freq == 0 and i > 0) and dist.get_rank() == 0:
            visualize_network_output(output_dict, input_dict, mode='train')

        if i % cfg.display_freq == 0 and dist.get_rank() == 0:
            gc.collect()
            print_inform = "({:d} / {:d}) lr: {:.6f}".format(i, len(train_loader), scheduler.get_last_lr()[0])
            for (k, v) in loss_dict.items():
                print_inform += " {}: {:.4f} ".format(k, v.item())
            print(print_inform)
        
        if (i+1) % 2000 == 0 and dist.get_rank() == 0:
            save_model(model, i, scheduler.get_last_lr(), optimizer)

    if cfg.exp_name == 'pre-training':
        if epoch % 1 == 0 and dist.get_rank() == 0:
            save_model(model, epoch, scheduler.get_last_lr(), optimizer)
    else:
        # if epoch % cfg.save_freq == 0 and dist.get_rank() == 0:
        if epoch % 1 == 0 and dist.get_rank() == 0:
            save_model(model, epoch, scheduler.get_last_lr(), optimizer)

    print('Training Loss: {}'.format(losses.avg))

def main():
    global lr
    # 获取本地 rank
    local_rank = cfg.local_rank
    print(local_rank)

    input_size_bucket = [512, 640, 800, 960]
    batch_size_bucket = [48, 48, 32, 24]
    data_root = "/xxx/TextBPN-Plus-Plus-main/data"
    valset = None
    trainset = AllTextJson(
        data_root=data_root,
        is_training=True,
        transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )
    print(f"Load data: {len(trainset)}")

    # 设置种子以确保可重复性
    seed = 42 + local_rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # g = torch.Generator(device='cuda')
    # g.manual_seed(seed)
    # print(torch.cuda.is_available())
    # print(cfg.device)

    
    # 创建分布式采样器
    train_sampler = DistributedSampler(trainset, num_replicas=dist.get_world_size(), 
                                            rank=local_rank, shuffle=True, seed=seed)

    # 创建数据加载器
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, 
                               sampler=train_sampler, pin_memory=True, generator=torch.Generator(device='cuda'))

    # Model
    model = TextNet(backbone=cfg.net, is_training=True)
    model = model.to(cfg.device)
    criterion = TextLoss()

    model = DDP(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank, find_unused_parameters=True)

    if cfg.resume:
        load_model(model, cfg.resume)

    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)

    if cfg.exp_name == 'pre-training':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)

    warmup_epochs = 5000  # 你可以根据需要调整 warmup 的 epoch 数
    base_lr = 0.0  # warmup 初始学习率
    final_lr = lr  # warmup 结束后的学习率
    scheduler = WarmupScheduler(optimizer, warmup_epochs, base_lr, final_lr, scheduler)

    # 确保所有进程同步
    dist.barrier()
    print('Start training TextBPN++.')
    scaler = GradScaler('cuda')
    for epoch in range(cfg.start_epoch, cfg.max_epoch + 1):
        train_sampler.set_epoch(epoch)
        train(model, train_loader, criterion, scheduler, optimizer, epoch, scaler, use_amp=cfg.use_amp, accum_grad_iters=cfg.accum_grad_iters)

    print('End.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')
    cfg.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(cfg.local_rank)
    print(cfg.local_rank)
    

    # main
    main()

    # Clean up the process group
    dist.destroy_process_group()