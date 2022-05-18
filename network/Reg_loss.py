# -*- coding: utf-8 -*-
# @Time    : 10/1/21
# @Author  : GXYM
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class PolyMatchingLoss(nn.Module):
    def __init__(self, pnum, device, loss_type="L1"):
        super(PolyMatchingLoss, self).__init__()

        self.pnum = pnum
        self.device = device
        self.loss_type = loss_type
        self.smooth_L1 = F.smooth_l1_loss
        self.L2_loss = torch.nn.MSELoss(reduce=False, size_average=False)

        batch_size = 1
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx

        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(device)
        self.feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), 2).detach()
        print(self.feature_id.shape)

    def match_loss(self, pred, gt):
        batch_size = pred.shape[0]
        feature_id = self.feature_id.expand(batch_size, self.feature_id.size(1), 2)

        gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, self.pnum, self.pnum, 2)
        pred_expand = pred.unsqueeze(1)

        if self.loss_type == "L2":
            dis = self.L2_loss(pred_expand, gt_expand)
            dis = dis.sum(3).sqrt().mean(2)
        elif self.loss_type == "L1":
            dis = self.smooth_L1(pred_expand, gt_expand, reduction='none')
            dis = dis.sum(3).mean(2)

        min_dis, min_id = torch.min(dis, dim=1, keepdim=True)

        return min_dis

    def forward(self, pred_list, gt):
        loss = torch.tensor(0.)
        for pred in pred_list:
            loss += torch.mean(self.match_loss(pred, gt))

        return loss / torch.tensor(len(pred_list))

        # los = []
        # for pred in pred_list:
        #     los.append(self.match_loss(pred, gt))
        #
        # los_b = torch.tensor(0.)
        # loss_c = torch.tensor(0.)
        # for i, _ in enumerate(los):
        #     los_b += torch.mean(los[i])
        #     loss_c += (torch.mean(torch.clamp(los[i] - los[i - 1], min=0.0)) if i > 0 else torch.tensor(0.))
        # loss = los_b / torch.tensor(len(los)) + 0.5*loss_c / torch.tensor(len(los)-1)
        #
        # return loss


class AttentionLoss(nn.Module):
    def __init__(self, beta=4, gamma=0.5):
        super(AttentionLoss, self).__init__()

        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, gt):
        num_pos = torch.sum(gt)
        num_neg = torch.sum(1 - gt)
        alpha = num_neg / (num_pos + num_neg)
        edge_beta = torch.pow(self.beta, torch.pow(1 - pred, self.gamma))
        bg_beta = torch.pow(self.beta, torch.pow(pred, self.gamma))

        loss = 0
        loss = loss - alpha * edge_beta * torch.log(pred) * gt
        loss = loss - (1 - alpha) * bg_beta * torch.log(1 - pred) * (1 - gt)
        return torch.mean(loss)


class GeoCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GeoCrossEntropyLoss, self).__init__()

    def forward(self, output, target, poly):
        output = torch.nn.functional.softmax(output, dim=1)
        output = torch.log(torch.clamp(output, min=1e-4))
        poly = poly.view(poly.size(0), 4, poly.size(1) // 4, 2)
        target = target[..., None, None].expand(poly.size(0), poly.size(1), 1, poly.size(3))
        target_poly = torch.gather(poly, 2, target)
        sigma = (poly[:, :, 0] - poly[:, :, 1]).pow(2).sum(2, keepdim=True)
        kernel = torch.exp(-(poly - target_poly).pow(2).sum(3) / (sigma / 3))
        loss = -(output * kernel.transpose(2, 1)).sum(1).mean()
        return loss


class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()

    def forward(self, ae, ind, ind_mask):
        """
        ae: [b, 1, h, w]
        ind: [b, max_objs, max_parts]
        ind_mask: [b, max_objs, max_parts]
        obj_mask: [b, max_objs]
        """
        # first index
        b, _, h, w = ae.shape
        b, max_objs, max_parts = ind.shape
        obj_mask = torch.sum(ind_mask, dim=2) != 0

        ae = ae.view(b, h * w, 1)
        seed_ind = ind.view(b, max_objs * max_parts, 1)
        tag = ae.gather(1, seed_ind).view(b, max_objs, max_parts)

        # compute the mean
        tag_mean = tag * ind_mask
        tag_mean = tag_mean.sum(2) / (ind_mask.sum(2) + 1e-4)

        # pull ae of the same object to their mean
        pull_dist = (tag - tag_mean.unsqueeze(2)).pow(2) * ind_mask
        obj_num = obj_mask.sum(dim=1).float()
        pull = (pull_dist.sum(dim=(1, 2)) / (obj_num + 1e-4)).sum()
        pull /= b

        # push away the mean of different objects
        push_dist = torch.abs(tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2))
        push_dist = 1 - push_dist
        push_dist = nn.functional.relu(push_dist, inplace=True)
        obj_mask = (obj_mask.unsqueeze(1) + obj_mask.unsqueeze(2)) == 2
        push_dist = push_dist * obj_mask.float()
        push = ((push_dist.sum(dim=(1, 2)) - obj_num) / (obj_num * (obj_num - 1) + 1e-4)).sum()
        push /= b
        return pull, push


def smooth_l1_loss(inputs, target, sigma=9.0):
    try:
        diff = torch.abs(inputs - target)
        less_one = (diff < 1.0 / sigma).float()
        loss = less_one * 0.5 * diff ** 2 * sigma \
               + torch.abs(torch.tensor(1.0) - less_one) * (diff - 0.5 / sigma)
        loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
    except Exception as e:
        print('RPN_REGR_Loss Exception:', e)
        loss = torch.tensor(0.0)

    return loss


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)