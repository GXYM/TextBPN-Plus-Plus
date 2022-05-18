###################################################################
# File Name: GCN.py
# Author: S.X.Zhang
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.bn0 = nn.BatchNorm1d(in_dim, affine=False)

        self.conv1 = GraphConv(in_dim, 256, MeanAggregator)
        self.conv2 = GraphConv(256, 1024, MeanAggregator)
        self.conv3 = GraphConv(1024, 512, MeanAggregator)
        self.conv4 = GraphConv(512, out_dim, MeanAggregator)

        self.prediction = nn.Sequential(
            nn.Conv1d(out_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1))

    def forward(self, x, A):
        x = self.bn0(x)
        x = x.permute(0, 2, 1)
        b, n, c = x.shape
        A = A.expand(b, n, n)

        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)

        x = x.permute(0, 2, 1)
        pred = self.prediction(x)

        return pred
