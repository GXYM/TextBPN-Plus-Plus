###################################################################
# File Name: GCN.py
# Author: S.X.Zhang
###################################################################

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable
import numpy as np
from cfglib.config import config as cfg


class Positional_encoding(nn.Module):
    def __init__(self, PE_size, n_position=200):
        super(Positional_encoding, self).__init__()
        self.PE_size = PE_size
        self.n_position = n_position
        self.register_buffer('pos_table', self.get_encoding_table(n_position, PE_size))

    def get_encoding_table(self, n_position, PE_size):
        position_table = np.array(
            [[pos / np.power(10000, 2. * i / self.PE_size) for i in range(self.PE_size)] for pos in range(n_position)])
        position_table[:, 0::2] = np.sin(position_table[:, 0::2])
        position_table[:, 1::2] = np.cos(position_table[:, 1::2])
        return torch.FloatTensor(position_table).unsqueeze(0)

    def forward(self, inputs):
        return inputs + self.pos_table[:, :inputs.size(1), :].clone().detach()


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_size, attention_size,
                 drop_rate, future_blind=True, query_mask=False, if_resi=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.drop_rate = drop_rate
        self.future_blind = future_blind
        
        self.Q_proj = nn.Sequential(nn.Linear(self.embedding_size, self.attention_size), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.embedding_size, self.attention_size), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.embedding_size, self.attention_size), nn.ReLU())

        self.drop_out = nn.Dropout(p=self.drop_rate)
        self.layer_norm = nn.LayerNorm(self.attention_size)
        self.if_resi = if_resi

    def forward(self, query, key, value):
        q = self.Q_proj(query)
        k = self.K_proj(key)
        v = self.V_proj(value)

        q_ = torch.cat(torch.chunk(q, self.num_heads, dim=2), dim=0)
        k_ = torch.cat(torch.chunk(k, self.num_heads, dim=2), dim=0)
        v_ = torch.cat(torch.chunk(v, self.num_heads, dim=2), dim=0)

        outputs = torch.bmm(q_, k_.permute(0, 2, 1))
        outputs = outputs / (k_.size()[-1] ** 0.5)

        # key mask

        # future mask
        if self.future_blind:
            diag_vals = torch.ones_like(outputs[0, :, :]).to(cfg.device)
            tril = torch.tril(diag_vals, diagonal=0)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))  # (h*N,T_q,T_k)
            padding = Variable(torch.ones_like(masks).to(cfg.device) * (-2 ** 32 + 1))
            condition = masks.eq(0)
            outputs = torch.where(condition, padding, outputs)

        outputs = F.softmax(outputs, dim=-1)
        # if self.future_blind==True:a
        #     print(outputs[0])
        outputs = self.drop_out(outputs)

        outputs = torch.bmm(outputs, v_)
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # N,T_q,C

        if self.if_resi:
            # outputs += query
            outputs += q
        else:
            outputs = outputs
        outputs = self.layer_norm(outputs)

        return outputs


class FeedForward(nn.Module):
    def __init__(self, in_channel, FFN_channel, if_resi=True):
        super(FeedForward, self).__init__()
        """
        1024 2048
        """
        output_channel = (FFN_channel, in_channel)
        self.fc1 = nn.Sequential(nn.Linear(in_channel, output_channel[0]), nn.ReLU())
        self.fc2 = nn.Linear(output_channel[0], output_channel[1])
        self.layer_norm = nn.LayerNorm(in_channel)
        self.if_resi = if_resi

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.fc2(outputs)
        if self.if_resi:
            outputs += inputs
        else:
            outputs = outputs
        outputs = self.layer_norm(outputs)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, out_dim, num_heads, embedding_size, attention_size,
                 dim_feedforward=1024, drop_rate=0.1, if_resi=True, block_nums=3):
        super(TransformerLayer, self).__init__()
        self.block_nums = block_nums
        self.if_resi = if_resi
        for i in range(self.block_nums):
            self.__setattr__('MHA_self_%d' % i, MultiHeadAttention(num_heads, embedding_size, attention_size,
                                                                   drop_rate, future_blind=False, if_resi=if_resi))
            self.__setattr__('FFN_%d' % i, FeedForward(out_dim, dim_feedforward, if_resi=if_resi))

    def forward(self, query):
        outputs = None
        for i in range(self.block_nums):
            outputs = self.__getattr__('MHA_self_%d' % i)(query, query, query)
            outputs = self.__getattr__('FFN_%d' % i)(outputs)
        return outputs


class Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads=8,
                 dim_feedforward=1024, drop_rate=0.1, if_resi=False, block_nums=3):
        super().__init__()

        self.bn0 = nn.BatchNorm1d(in_dim, affine=False)
        self.conv1 = nn.Conv1d(in_dim, out_dim, 1, dilation=1)

        embed_dim = in_dim
        # self.pos_embedding = Positional_encoding(embed_dim)
        self.transformer = TransformerLayer(out_dim, num_heads, embedding_size=embed_dim,
                                            attention_size=out_dim, dim_feedforward=dim_feedforward,
                                            drop_rate=drop_rate, if_resi=if_resi, block_nums=block_nums)

        self.prediction = nn.Sequential(
            nn.Conv1d(out_dim*2, 128, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Conv1d(64, 2, 1))

    def forward(self, x, adj):
        x = self.bn0(x)

        x1 = x.permute(0, 2, 1)
        x1 = self.transformer(x1)
        x1 = x1.permute(0, 2, 1)

        x = torch.cat([x1, self.conv1(x)], dim=1)
        # x = x1+self.conv1(x)
        pred = self.prediction(x)

        return pred



