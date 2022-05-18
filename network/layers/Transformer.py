###################################################################
# File Name: GCN.py
# Author: S.X.Zhang
###################################################################
import torch
from torch import nn, Tensor
import numpy as np
from cfglib.config import config as cfg


class Positional_encoding(nn.Module):
    def __init__(self, PE_size, n_position=256):
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
    def __init__(self, num_heads, embed_dim, dropout=0.1, if_resi=True):
        super(MultiHeadAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.MultiheadAttention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.Q_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.if_resi = if_resi

    def forward(self, inputs):
        query = self.layer_norm(inputs)
        q = self.Q_proj(query)
        k = self.K_proj(query)
        v = self.V_proj(query)
        attn_output, attn_output_weights = self.MultiheadAttention(q, k, v)
        if self.if_resi:
            attn_output += inputs
        else:
            attn_output = attn_output

        return attn_output


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
        outputs = self.layer_norm(inputs)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        if self.if_resi:
            outputs += inputs
        else:
            outputs = outputs
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, out_dim, in_dim, num_heads, attention_size,
                 dim_feedforward=1024, drop_rate=0.1, if_resi=True, block_nums=3):
        super(TransformerLayer, self).__init__()
        self.block_nums = block_nums
        self.if_resi = if_resi
        self.linear = nn.Linear(in_dim, attention_size)
        for i in range(self.block_nums):
            self.__setattr__('MHA_self_%d' % i, MultiHeadAttention(num_heads, attention_size,
                                                                   dropout=drop_rate, if_resi=if_resi))
            self.__setattr__('FFN_%d' % i, FeedForward(out_dim, dim_feedforward, if_resi=if_resi))

    def forward(self, query):
        inputs = self.linear(query)
        # outputs = inputs
        for i in range(self.block_nums):
            outputs = self.__getattr__('MHA_self_%d' % i)(inputs)
            outputs = self.__getattr__('FFN_%d' % i)(outputs)
            if self.if_resi:
                inputs = inputs+outputs
            else:
                inputs = outputs
        # outputs = inputs
        return inputs


class Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads=8,
                 dim_feedforward=1024, drop_rate=0.1, if_resi=False, block_nums=3):
        super().__init__()

        self.bn0 = nn.BatchNorm1d(in_dim, affine=False)
        self.conv1 = nn.Conv1d(in_dim, out_dim, 1, dilation=1)

        # self.pos_embedding = Positional_encoding(in_dim)
        self.transformer = TransformerLayer(out_dim, in_dim, num_heads, attention_size=out_dim,
                                            dim_feedforward=dim_feedforward, drop_rate=drop_rate,
                                            if_resi=if_resi, block_nums=block_nums)

        self.prediction = nn.Sequential(
            nn.Conv1d(2*out_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Conv1d(64, 2, 1))

    def forward(self, x, adj):
        x = self.bn0(x)

        x1 = x.permute(0, 2, 1)
        # x1 = self.pos_embedding(x1)
        x1 = self.transformer(x1)
        x1 = x1.permute(0, 2, 1)

        x = torch.cat([x1, self.conv1(x)], dim=1)
        # x = x1+self.conv1(x)
        pred = self.prediction(x)

        return pred



