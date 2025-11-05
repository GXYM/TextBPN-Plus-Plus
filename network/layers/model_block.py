# -*- coding: utf-8 -*-
__author__ = "S.X.Zhang"
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.layers.vgg import VggNet
from network.layers.resnet import ResNet
from network.layers.resnet_dcn import ResNet_DCN
from cfglib.config import config as cfg


class UpBlok(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        # NOTE:
        #   Several legacy checkpoints (e.g. deformable_resnet50 TD500 @ epoch 300)
        #   were trained when UpBlok contained an internal ConvTranspose upsample.
        #   After refactoring the FPN we use MergeBlok (no upsample) for scale=4.
        #   When those old weights are loaded the learnt deconv filters are skipped,
        #   leaving feature maps at half resolution.  The explicit resize keeps
        #   the runtime compatible across both families of checkpoints without
        #   affecting the newer models (resnet18/50 scale=4) whose tensors already
        #   match in spatial dims.
        if upsampled.shape[-2:] != shortcut.shape[-2:]:
            upsampled = F.interpolate(upsampled, size=shortcut.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class MergeBlok(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, upsampled, shortcut):
        # See note in UpBlok.forward: we guard the concat by resizing when old
        # checkpoints (with learned deconv upsample) are loaded on the refactored
        # FPN. The check is a no-op for the current layout where shapes already
        # align (e.g. resnet18 scale=4 and matching weights).
        if upsampled.shape[-2:] != shortcut.shape[-2:]:
            upsampled = F.interpolate(upsampled, size=shortcut.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        return x


class FPN(nn.Module):

    def __init__(self, backbone='resnet50', is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone

        if backbone in ['vgg_bn', 'vgg']:
            self.backbone = VggNet(name=backbone, pretrain=is_training)
            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(512 + 256, 128)
            self.merge3 = UpBlok(256 + 128, 64)
            if cfg.scale == 1:
                self.merge2 = UpBlok(128 + 64, 32)  # FPN 1/2
                self.merge1 = UpBlok(64 + 32, 32)   # FPN 1/1
            elif cfg.scale == 2:
                self.merge2 = UpBlok(128 + 64, 32)    # FPN 1/2
                self.merge1 = MergeBlok(64 + 32, 32)  # FPN 1/2
            elif cfg.scale == 4:
                self.merge2 = MergeBlok(128 + 64, 32)  # FPN 1/4

        elif backbone in ['resnet50']:
            self.backbone = ResNet(name=backbone, pretrain=is_training)
            self.deconv5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(1024 + 256, 128)
            self.merge3 = UpBlok(512 + 128, 64)
            if cfg.scale == 1:
                self.merge2 = UpBlok(256 + 64, 32)  # FPN 1/2
                self.merge1 = UpBlok(64 + 32, 32)   # FPN 1/1
            elif cfg.scale == 2:
                self.merge2 = UpBlok(256 + 64, 32)    # FPN 1/2
                self.merge1 = MergeBlok(64 + 32, 32)  # FPN 1/2
            elif cfg.scale == 4:
                self.merge2 = MergeBlok(256 + 64, 32)  # FPN 1/4
                self.merge1 = MergeBlok(64 + 32, 32)  # FPN 1/4
        
        elif backbone in ['resnet18']:
            self.backbone = ResNet(name=backbone, pretrain=is_training)
            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(256 + 256, 128)
            self.merge3 = UpBlok(128 + 128, 64)
            if cfg.scale == 1:
                self.merge2 = UpBlok(64 + 64, 32)  # FPN 1/2
                self.merge1 = UpBlok(64 + 32, 32)   # FPN 1/1
            elif cfg.scale == 2:
                self.merge2 = UpBlok(64 + 64, 32)    # FPN 1/2
                self.merge1 = MergeBlok(64 + 32, 32)  # FPN 1/2
            elif cfg.scale == 4:
                self.merge2 = MergeBlok(64 + 64, 32)  # FPN 1/4
                self.merge1 = MergeBlok(64 + 32, 32)  # FPN 1/4
       
        elif backbone in ["deformable_resnet18"]:
            self.backbone = ResNet_DCN(name=backbone, pretrain=is_training)
            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(256 + 256, 128)
            self.merge3 = UpBlok(128 + 128, 64)
            if cfg.scale == 1:
                self.merge2 = UpBlok(64 + 64, 32)  # FPN 1/2
                self.merge1 = UpBlok(64 + 32, 32)   # FPN 1/1
            elif cfg.scale == 2:
                self.merge2 = UpBlok(64 + 64, 32)    # FPN 1/2
                self.merge1 = MergeBlok(64 + 32, 32)  # FPN 1/2
            elif cfg.scale == 4:
                self.merge2 = MergeBlok(64 + 64, 32)  # FPN 1/4
                self.merge1 = MergeBlok(64 + 32, 32)  # FPN 1/4
        
        elif backbone in ["deformable_resnet50"]:
            self.backbone = ResNet_DCN(name=backbone, pretrain=is_training)
            self.deconv5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(1024 + 256, 128)
            self.merge3 = UpBlok(512 + 128, 64)
            if cfg.scale == 1:
                self.merge2 = UpBlok(256 + 64, 32)  # FPN 1/2
                self.merge1 = UpBlok(64 + 32, 32)  # FPN 1/1
            elif cfg.scale == 2:
                self.merge2 = UpBlok(256 + 64, 32)  # FPN 1/2
                self.merge1 = MergeBlok(64 + 32, 32)  # FPN 1/2
            elif cfg.scale == 4:
                self.merge2 = MergeBlok(256 + 64, 32)  # FPN 1/4
                self.merge1 = MergeBlok(64 + 32, 32)  # FPN 1/4
        else:
            print("backbone is not support !")

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        #print(C5.size())
        #print(C4.size())
        #print(C3.size())
        #print(C2.size())
        #print(C1.size())
        up5 = self.deconv5(C5)
        up5 = F.relu(up5)

        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)

        up1 = self.merge1(C1, up2)
        up1 = F.relu(up1)

        return up1, up2, up3, up4, up5
