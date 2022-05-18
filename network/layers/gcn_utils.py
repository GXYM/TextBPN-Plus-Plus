# -*- coding: utf-8 -*-
__author__ = "S.X.Zhang"
import torch
import numpy as np
import cv2
import torch.nn as nn
from torch.autograd import Variable


def normalize_adj(A, type="AD"):
    if type == "DAD":
        A = A + np.eye(A.shape[0])  # A=A+I
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(d_inv).transpose().dot(d_inv)  # L = D^-1/2 A D^-1/2
        G = torch.from_numpy(G)
    elif type == "AD":
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        G = A.div(D)  # L= A/D
    else:
        A = A + np.eye(A.shape[0])  # A=A+I
        D = A.sum(1, keepdim=True)
        D = np.diag(D)
        G = torch.from_numpy(D - A)  # L = D-A
    return G


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A,BT)
    SqA = A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED


def get_center_feature(cnn_feature, img_poly, ind, h, w):
    batch_size = cnn_feature.size(0)
    for i in range(batch_size):
        poly = img_poly[ind == i].cpu().numpy()
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, poly.astype(np.int32), color=(1,))
    return None


def get_node_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone().float()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        gcn_feature[ind == i] = torch.nn.functional.grid_sample(cnn_feature[i:i + 1], poly)[0].permute(1, 0, 2)
    return gcn_feature


def get_adj_mat(n_adj, n_nodes):
    a = np.zeros([n_nodes, n_nodes], dtype=np.float)

    for i in range(n_nodes):
        for j in range(-n_adj // 2, n_adj // 2 + 1):
            if j != 0:
                a[i][(i + j) % n_nodes] = 1
                a[(i + j) % n_nodes][i] = 1
    return a


def get_adj_ind(n_adj, n_nodes, device):
    ind = torch.tensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0]).long()
    ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
    return ind.to(device)


def coord_embedding(b, w, h, device):
    x_range = torch.linspace(0, 1, w, device=device)
    y_range = torch.linspace(0, 1, h, device=device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([b, 1, -1, -1])
    x = x.expand([b, 1, -1, -1])
    coord_map = torch.cat([x, y], 1)

    return coord_map


def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    # x_max = torch.max(img_poly[..., 0], dim=-1)[0]
    # y_max = torch.max(img_poly[..., 1], dim=-1)[0]
    # h, w = y_max - y_min + 1, x_max - x_min + 1
    # long_side = torch.max(h, w)
    # can_poly = can_poly / long_side[..., None, None]
    return can_poly
