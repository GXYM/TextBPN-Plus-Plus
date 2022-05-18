# -*- coding: utf-8 -*-
# @Time    : 10/1/21
# @Author  : GXYM
import torch
import torch.nn as nn
from network.layers.model_block import FPN
from cfglib.config import config as cfg
import numpy as np
from network.layers.CircConv import DeepSnake
from network.layers.GCN import GCN
from network.layers.RNN import RNN
from network.layers.Adaptive_Deformation import AdaptiveDeformation
# from network.layers.Transformer_old import Transformer_old
from network.layers.Transformer import Transformer
import cv2
from util.misc import get_sample_point, fill_hole
from network.layers.gcn_utils import get_node_feature, \
    get_adj_mat, get_adj_ind, coord_embedding, normalize_adj
import torch.nn.functional as F
import time


class Evolution(nn.Module):
    def __init__(self, node_num, adj_num, is_training=True, device=None, model="snake"):
        super(Evolution, self).__init__()
        self.node_num = node_num
        self.adj_num = adj_num
        self.device = device
        self.is_training = is_training
        self.clip_dis = 16

        self.iter = 3
        if model == "gcn":
            self.adj = get_adj_mat(self.adj_num, self.node_num)
            self.adj = normalize_adj(self.adj, type="DAD").float().to(self.device)
            for i in range(self.iter):
                evolve_gcn = GCN(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "rnn":
            self.adj = None
            for i in range(self.iter):
                evolve_gcn = RNN(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "AD":
            self.adj = get_adj_mat(self.adj_num, self.node_num)
            self.adj = normalize_adj(self.adj, type="DAD").float().to(self.device)
            for i in range(self.iter):
                evolve_gcn = AdaptiveDeformation(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        # elif model == "BT_old":
        #     self.adj = None
        #     for i in range(self.iter):
        #         evolve_gcn = Transformer_old(36, 512, num_heads=8,
        #                                  dim_feedforward=2048, drop_rate=0.0, if_resi=True, block_nums=4)
        #         self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "BT":
            self.adj = None
            for i in range(self.iter):
                evolve_gcn = Transformer(36, 128, num_heads=8,
                                         dim_feedforward=1024, drop_rate=0.0, if_resi=True, block_nums=3)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        else:
            self.adj = get_adj_ind(self.adj_num, self.node_num, self.device)
            for i in range(self.iter):
                evolve_gcn = DeepSnake(state_dim=128, feature_dim=36, conv_type='dgrid')
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_boundary_proposal(input=None, seg_preds=None, switch="gt"):

        if switch == "gt":
            inds = torch.where(input['ignore_tags'] > 0)
            # if len(inds[0]) > 320:
            #    inds = (inds[0][:320], inds[1][:320])
            init_polys = input['proposal_points'][inds]
        else:
            tr_masks = input['tr_mask'].cpu().numpy()
            tcl_masks = seg_preds[:, 0, :, :].detach().cpu().numpy() > cfg.threshold
            inds = []
            init_polys = []
            for bid, tcl_mask in enumerate(tcl_masks):
                ret, labels = cv2.connectedComponents(tcl_mask.astype(np.uint8), connectivity=8)
                for idx in range(1, ret):
                    text_mask = labels == idx
                    ist_id = int(np.sum(text_mask*tr_masks[bid])/np.sum(text_mask))-1
                    inds.append([bid, ist_id])
                    poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor)
                    init_polys.append(poly)
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device)

        return init_polys, inds, None

    def get_boundary_proposal_eval(self, input=None, seg_preds=None):

        # if cfg.scale > 1:
        #     seg_preds = F.interpolate(seg_preds, scale_factor=cfg.scale, mode='bilinear')
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, ].detach().cpu().numpy()

        inds = []
        init_polys = []
        confidences = []
        for bid, dis_pred in enumerate(dis_preds):
            # # dis_mask = (dis_pred / np.max(dis_pred)) > cfg.dis_threshold
            dis_mask = dis_pred > cfg.dis_threshold
            # dis_mask = fill_hole(dis_mask)
            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = round(cls_preds[bid][text_mask].mean(), 3)
                # 50 for MLT2017 and ArT (or DCN is used in backone); else is all 150;
                # just can set to 50, which has little effect on the performance
                if np.sum(text_mask) < 50/(cfg.scale*cfg.scale) or confidence < cfg.cls_threshold:
                    continue
                confidences.append(confidence)
                inds.append([bid, 0])
                
                poly = get_sample_point(text_mask, cfg.num_points,
                                        cfg.approx_factor, scales=np.array([cfg.scale, cfg.scale]))
                init_polys.append(poly)

        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
        else:
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
            inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)

        return init_polys, inds, confidences

    def evolve_poly(self, snake, cnn_feature, i_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2)*cfg.scale, cnn_feature.size(3)*cfg.scale
        node_feats = get_node_feature(cnn_feature, i_it_poly, ind, h, w)
        i_poly = i_it_poly + torch.clamp(snake(node_feats, self.adj).permute(0, 2, 1), -self.clip_dis, self.clip_dis)
        if self.is_training:
            i_poly = torch.clamp(i_poly, 0, w-1)
        else:
            i_poly[:, :, 0] = torch.clamp(i_poly[:, :, 0], 0, w - 1)
            i_poly[:, :, 1] = torch.clamp(i_poly[:, :, 1], 0, h - 1)
        return i_poly

    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt"):
        if self.is_training:
            init_polys, inds, confidences = self.get_boundary_proposal(input=input, seg_preds=seg_preds, switch=switch)
            # TODO sample fix number
        else:
            init_polys, inds, confidences = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            if init_polys.shape[0] == 0:
                return [init_polys for i in range(self.iter+1)], inds, confidences

        py_preds = [init_polys, ]
        for i in range(self.iter):
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            init_polys = self.evolve_poly(evolve_gcn, embed_feature, init_polys, inds[0])
            py_preds.append(init_polys)

        return py_preds, inds, confidences


class TextNet(nn.Module):

    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, is_training=(not cfg.resume and is_training))

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )
        self.BPN = Evolution(cfg.num_points, adj_num=4,
                             is_training=is_training, device=cfg.device, model="BT")

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path, map_location=torch.device(cfg.device))
        self.load_state_dict(state_dict['model'], strict=(not self.is_training))

    def forward(self, input_dict, test_speed=False):
        output = {}
        b, c, h, w = input_dict["img"].shape
        if self.is_training or cfg.exp_name in ['ArT', 'MLT2017', "MLT2019"] or test_speed:
            image = input_dict["img"]
        else:
            image = torch.zeros((b, c, cfg.test_size[1], cfg.test_size[1]), dtype=torch.float32).to(cfg.device)
            image[:, :, :h, :w] = input_dict["img"][:, :, :, :]

        up1, _, _, _, _ = self.fpn(image)
        up1 = up1[:, :, :h // cfg.scale, :w // cfg.scale]

        preds = self.seg_head(up1)
        fy_preds = torch.cat([torch.sigmoid(preds[:, 0:2, :, :]), preds[:, 2:4, :, :]], dim=1)
        cnn_feats = torch.cat([up1, fy_preds], dim=1)

        py_preds, inds, confidences = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")
        
        output["fy_preds"] = fy_preds
        output["py_preds"] = py_preds
        output["inds"] = inds
        output["confidences"] = confidences

        return output
