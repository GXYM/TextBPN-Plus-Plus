# -*- coding: utf-8 -*-
# @Time    : 10/1/21
# @Author  : GXYM
import torch
import torch.nn as nn
from cfglib.config import config as cfg
from network.Seg_loss import SegmentLoss
from network.Reg_loss import PolyMatchingLoss


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)
        self.BCE_loss = torch.nn.BCELoss(reduce=False, size_average=False)
        self.PolyMatchingLoss = PolyMatchingLoss(cfg.num_points, cfg.device)
        self.KL_loss = torch.nn.KLDivLoss(reduce=False, size_average=False)

    @staticmethod
    def single_image_loss(pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1)) * 0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        eps = 0.001
        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= eps)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= eps)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < eps)]) < 3 * positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < eps)])
                    average_number += len(pre_loss[i][(loss_label[i] < eps)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < eps)], 3 * positive_pixel)[0])
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 100)[0])
                average_number += 100
                sum_loss += nega_loss
            # sum_loss += loss/average_number

        return sum_loss/batch_size

    def cls_ohem(self, predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()

        n_pos = pos.float().sum()

        if n_pos.item() > 0:
            loss_pos = self.BCE_loss(predict[pos], target[pos]).sum()
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.)
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    @staticmethod
    def loss_calc_flux(pred_flux, gt_flux, weight_matrix, mask, train_mask):

        # norm loss
        gt_flux = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1).unsqueeze(1) + 1e-9)
        norm_loss = weight_matrix * torch.sum((pred_flux - gt_flux) ** 2, dim=1)*train_mask
        norm_loss = norm_loss.sum(-1).mean()

        # angle loss
        mask = train_mask * mask
        pred_flux = 0.999999 * pred_flux / (pred_flux.norm(p=2, dim=1).unsqueeze(1) + 1e-9)
        # angle_loss = weight_matrix * (torch.acos(torch.sum(pred_flux * gt_flux, dim=1))) ** 2
        # angle_loss = angle_loss.sum(-1).mean()
        angle_loss = (1 - torch.cosine_similarity(pred_flux, gt_flux, dim=1))
        angle_loss = angle_loss[mask].mean()

        return norm_loss, angle_loss

    def forward(self, input_dict, output_dict, eps=None):
        """
          calculate boundary proposal network loss
        """
        # tr_mask = tr_mask.permute(0, 3, 1, 2).contiguous()

        fy_preds = output_dict["fy_preds"]
        py_preds = output_dict["py_preds"]
        inds = output_dict["inds"]

        train_mask = input_dict['train_mask']
        tr_mask = input_dict['tr_mask'] > 0
        distance_field = input_dict['distance_field']
        direction_field = input_dict['direction_field']
        weight_matrix = input_dict['weight_matrix']
        gt_tags = input_dict['gt_points']

        # pixel class loss
        cls_loss = self.cls_ohem(fy_preds[:, 0, :, :], tr_mask.float(), train_mask.bool())

        # distance field loss
        dis_loss = self.MSE_loss(fy_preds[:, 1, :, :], distance_field)
        dis_loss = torch.mul(dis_loss, train_mask.float())
        dis_loss = self.single_image_loss(dis_loss, distance_field)

        # direction field loss
        norm_loss, angle_loss = self.loss_calc_flux(fy_preds[:, 2:4, :, :],
                                                    direction_field, weight_matrix, tr_mask, train_mask)

        # boundary point loss
        point_loss = self.PolyMatchingLoss(py_preds, gt_tags[inds])

        if eps is None:
            loss_b = 0.05*point_loss
            loss = cls_loss + 3.0*dis_loss + norm_loss + angle_loss + loss_b
        else:
            loss_b = 0.1*(torch.sigmoid(torch.tensor((eps - cfg.max_epoch)/cfg.max_epoch))) * point_loss
            loss = cls_loss + 3.0*dis_loss + norm_loss + angle_loss + loss_b

        loss_dict = {
            'total_loss': loss,
            'cls_loss': cls_loss,
            'distance loss': 3.0*dis_loss,
            'dir_loss': norm_loss + angle_loss,
            'point_loss': loss_b,
            'norm_loss': norm_loss,
            'angle_loss': angle_loss,

        }

        return loss_dict

